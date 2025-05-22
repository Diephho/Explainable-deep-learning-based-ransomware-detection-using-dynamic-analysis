# sandbox_runner.py
import requests
import time
import sys
import os
import json
from model import model as build_model
from config import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX, SEQ_LEN
import numpy as np
from data_utils import prepare_sequences
from explain import build_id2token, lime_explain_instance
import matplotlib.pyplot as plt

CUCKOO_API = "http://192.168.141.128:8090"
def build_id2token(token2id):
    id2token = {int(idx): tok for tok, idx in token2id.items()}
    if 0 not in id2token:
        id2token[0] = "<PAD>"
    return id2token

def submit_sample(file_path):
    with open(file_path, 'rb') as f:
        r = requests.post(f"{CUCKOO_API}/tasks/create/file", files={'file': f})
    r.raise_for_status()
    task_id = r.json()['task_id']
    print(f"[+] Submitted. Task ID: {task_id}")
    return task_id

def wait_for_report(task_id):
    while True:
        r = requests.get(f"{CUCKOO_API}/tasks/view/{task_id}")
        r.raise_for_status()
        status = r.json()['task']['status']
        if status == "reported":
            print("[+] Analysis done.")
            break
        print(f"[-] Waiting for task {task_id}... (status: {status})")
        time.sleep(5)

def download_report(task_id, save_to):
    r = requests.get(f"{CUCKOO_API}/tasks/report/{task_id}")
    r.raise_for_status()
    with open(save_to, 'w') as f:
        f.write(r.text)
    print(f"[+] Report downloaded to {save_to}")

# Hàm trích xuất dll, api, mutex
def extract_fields(report_path, attribute_path):
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)

        dlls = []
        apis = []
        mutexes = []

        # Trích xuất các trường nếu tồn tại
        if 'behavior' in data:
            if 'summary' in data['behavior']:
                summary = data['behavior']['summary']
                if 'mutex' in summary:
                    mutexes = summary.get('mutex', [])
                
                if 'dll_loaded' in summary:
                    dlls = summary.get('dll_loaded', [])

            if 'processes' in data['behavior']:
                for process in data['behavior']['processes']:
                    if 'calls' in process:
                        for call in process['calls']:
                            if 'api' in call:
                                apis.append(call['api'])
            
        
        attributes = {
            "dlls": dlls[:10],
            "apis": apis[:500],
            "mutexes": mutexes[:10]
        }
        report_file = os.path.splitext(os.path.basename(report_path))[0]
        with open(attribute_path, 'w') as f:
            json.dump(attributes, f, indent=4)

        print(f"[+] Đã ghi thông tin vào: {attribute_path}")


    except Exception as e:
        print(f"[!] Lỗi đọc hoặc phân tích report {report_path}: {e}")

def check_valid(report_extract_path):
    if not os.path.isfile(report_extract_path):
        print(f"❌ File không tồn tại: {report_extract_path}")
        return False

    try:
        with open(report_extract_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("❌ File không phải là JSON hợp lệ.")
        return False

    # Kiểm tra các trường bắt buộc
    required_fields = {"dlls", "apis", "mutexes"}
    data_keys = set(data.keys())

    if data_keys != required_fields:
        print(f"❌ File JSON phải chứa đúng các trường sau: {', '.join(required_fields)}")
        print(f"📌 Trường đang có: {', '.join(data_keys)}")
        return False

    print("✅ File JSON hợp lệ.")
    return True

def check_and_explain(report_extract_path):
    if check_valid(report_extract_path) != True:
        return "Invalid", 0.0, []

    with open('token2id.json', encoding='utf-8') as f:
        token2id = json.load(f)

    with open(report_extract_path, 'r', encoding='utf-8') as f:
        features = json.load(f)

    sequence = prepare_sequences([features], token2id)
    X_input = np.array(sequence)
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("./best_model.weights.h5")
    proba = cnn_model.predict(X_input)[0]

    label = "Ransomware" if proba[1] > 0.5 else "Benign"
    confidence = float(proba[1] if label == "Ransomware" else proba[0])

    tokens_no_pad = [x for x in sequence[0] if x != 0]
    id2token = build_id2token(token2id)
    lime_result = lime_explain_instance(cnn_model, tokens_no_pad, id2token)

    return label, confidence, lime_result

def plot_lime_top_5_5_to_file(local_lime, count, pred_label, pred_proba, save_path):
    # 1. Tách 2 nhóm
    pos = [(f, w) for f, w in local_lime if w > 0]
    neg = [(f, w) for f, w in local_lime if w < 0]

    # 2. Sort theo |weight| giảm dần và lấy top 5 mỗi nhóm
    pos = sorted(pos, key=lambda x: abs(x[1]), reverse=True)[:5]
    neg = sorted(neg, key=lambda x: abs(x[1]), reverse=True)[:5]

    # 3. Xen kẽ: bắt đầu bằng pos[0], rồi neg[0], v.v.
    interleaved = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            interleaved.append(pos[i])
        if i < len(neg):
            interleaved.append(neg[i])

    # 4. Tách lại để vẽ
    features, weights = zip(*interleaved)
    colors = ['green' if w > 0 else 'red' for w in weights]
    if isinstance(pred_proba, (list, tuple)):
        prob_display = pred_proba[1]
    else:
        prob_display = pred_proba
    # 5. Vẽ và lưu file
    plt.figure(figsize=(8, 4))
    plt.barh(features[::-1], weights[::-1], color=colors[::-1])  # Đảo để mạnh nhất nằm trên
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel('LIME Weight')
    plt.title(f'Sample {count}: {pred_label} (prob={prob_display:.5f})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()