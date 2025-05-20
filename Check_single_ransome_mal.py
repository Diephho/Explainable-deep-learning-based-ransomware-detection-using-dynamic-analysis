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
        return
    # 1. Load token2id
    with open('token2id_ransome_mal.json', encoding='utf-8') as f:
        token2id = json.load(f)
    # 2. Handle data in report_path
    reports=[]
    with open(report_extract_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
    reports.append(features)
    sequence = prepare_sequences(reports, token2id)
    X_input=np.array(sequence[0])
    # 3. Load Model
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("./best_model_ransome_mal.weights.h5")
    proba = cnn_model.predict(X_input)[0]
    # 4. Prediction
    label = "Ransomware" if proba[1] > 0.5 else "Malware"
    print(f"\nPrediction: {label}")
    # 5. Explain
    tokens_no_pad = [x for x in sequence if x != 0]
    id2token = build_id2token(token2id)
    lime_result = lime_explain_instance(cnn_model, tokens_no_pad, id2token)
    print("\nTop LIME features:")
    for token, weight in lime_result:
        print(f"{token}: {weight:.4f}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 Check_single_ransome_mal.py <file_path> <type>")
        print("\n<type> tùy vào loại file bạn muốn phân tích:")
        print("  [1] Phân tích file thực thi (.exe)")
        print("  [2] Phân tích file báo cáo (report file)")
        print("  [3] Phân tích file thuộc tính (attribute file)")
        sys.exit(1)

    if sys.argv[2] == 1:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            print("[-] File not found.")
            sys.exit(1)

        filename = os.path.basename(file_path)
        task_id = submit_sample(file_path)
        wait_for_report(task_id)

        os.makedirs("checkfile", exist_ok=True)
        os.makedirs("checkfile/reports", exist_ok=True)
        report_path = f"checkfile/reports/report_{filename}.json"
        download_report(task_id, report_path)
    else:
        if sys.argv[2] == 2:
            report_path = sys.argv[1]
            attribute_path = f"checkfile/reports/report_{filename}.json"
            extract_fields(report_path, attribute_path)
        else:
            attribute_path = sys.argv[1]
    check_and_explain(attribute_path)

if __name__ == "__main__":
    main()