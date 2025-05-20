# explain.py

import numpy as np
import json
from lime.lime_text import LimeTextExplainer
import shap
from data_utils import prepare_sequences
from model import model as build_model
from config_ransome_mal import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX, SEQ_LEN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

def build_id2token(token2id):
    id2token = {int(idx): tok for tok, idx in token2id.items()}
    if 0 not in id2token:
        id2token[0] = "<PAD>"
    return id2token


def lime_explain_instance(cnn_model, sequence, id2token, num_features=10):
    explainer = LimeTextExplainer(class_names=["malware","ransomware"], split_expression=r"\s+")
    def predict_proba(texts):
        seqs = []
        for t in texts:
            tokens = [int(x) for x in t.split()]
            # Padding hoặc cắt để độ dài đúng SEQ_LEN
            if len(tokens) < SEQ_LEN:
                tokens = tokens + [0] * (SEQ_LEN - len(tokens))
            else:
                tokens = tokens[:SEQ_LEN]
            seqs.append(tokens)
        X = np.array(seqs)
        probs = cnn_model.predict(X)
        return np.hstack([1 - probs, probs])
    text_input = " ".join(str(int(x)) for x in sequence)
    exp = explainer.explain_instance(text_input, predict_proba, num_features=num_features)
    return [(id2token[int(tok)], weight) for tok, weight in exp.as_list()]

def plot_top_shap_bar(top_tokens):
    features, weights = zip(*top_tokens)
    colors = ['blue' if w < 0 else 'red' for w in weights]  # ✅ Đúng như yêu cầu của Khang
    plt.figure(figsize=(8, 4))
    plt.barh(features, weights, color=colors)
    plt.xlabel('Mean SHAP Value')
    plt.title('Global SHAP (Top Features)')
    plt.tight_layout()
    plt.show()

def shap_explain_global_ransome(cnn_model, X_background, X_test, id2token, top_n=10):
    try:
        # Dùng GradientExplainer cho TF2.x
        explainer = shap.GradientExplainer(cnn_model, X_background)
        shap_vals = explainer.shap_values(X_test)
    except Exception:
        # Fallback: KernelExplainer (chậm hơn)
        f = lambda x: cnn_model.predict(x)
        explainer = shap.KernelExplainer(f, X_background[:50])
        shap_vals = explainer.shap_values(X_test[:20])
    
    # SHAP value cho lớp ransomware (index 1)
    sv = shap_vals[1]  # shape: (num_samples, seq_len)
    X_display = X_test[:sv.shape[0]]

    # 1. Tính mean |SHAP| theo từng token position
    mean_abs_shap = np.abs(sv).mean(axis=0)  # shape: (seq_len,)

    # 2. Ánh xạ token id -> tên
    token_ids = X_display[0]  # Lấy 1 sample để biết thứ tự token IDs ở mỗi position
    feature_names = [id2token.get(i, f"<UNK_{i}>") for i in token_ids]

    # 3. Ghép tên và SHAP vào, lấy top N
    token_shap_pairs = list(zip(feature_names, mean_abs_shap))
    token_shap_pairs.sort(key=lambda x: x[1], reverse=True)
    top_tokens = token_shap_pairs[:top_n]

    # 4. Hiển thị SHAP summary plot (tuỳ chọn)
    shap.summary_plot(sv, X_display, feature_names=feature_names)
    plot_top_shap_bar(top_tokens)
    return top_tokens

def shap_explain_global_malware(cnn_model, X_background, X_test, id2token, top_n=10):
    try:
        # Dùng GradientExplainer cho TF2.x
        explainer = shap.GradientExplainer(cnn_model, X_background)
        shap_vals = explainer.shap_values(X_test)
    except Exception:
        # Fallback: KernelExplainer (chậm hơn)
        f = lambda x: cnn_model.predict(x)
        explainer = shap.KernelExplainer(f, X_background[:50])
        shap_vals = explainer.shap_values(X_test[:20])
    
    # SHAP value cho lớp ransomware (index 1)
    sv = shap_vals[0]  # shape: (num_samples, seq_len)
    X_display = X_test[:sv.shape[0]]

    # 1. Tính mean |SHAP| theo từng token position
    mean_abs_shap = np.abs(sv).mean(axis=0)  # shape: (seq_len,)

    # 2. Ánh xạ token id -> tên
    token_ids = X_display[0]  # Lấy 1 sample để biết thứ tự token IDs ở mỗi position
    feature_names = [id2token.get(i, f"<UNK_{i}>") for i in token_ids]

    # 3. Ghép tên và SHAP vào, lấy top N
    token_shap_pairs = list(zip(feature_names, mean_abs_shap))
    token_shap_pairs.sort(key=lambda x: x[1], reverse=True)
    top_tokens = token_shap_pairs[:top_n]

    # 4. Hiển thị SHAP summary plot (tuỳ chọn)
    shap.summary_plot(sv, X_display, feature_names=feature_names)
    plot_top_shap_bar(top_tokens)
    return top_tokens

def plot_lime_top_5_5(local_lime, count, pred_label, pred_proba):
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

    # 5. Vẽ
    plt.figure(figsize=(8, 4))
    plt.barh(features[::-1], weights[::-1], color=colors[::-1])  # Đảo để mạnh nhất nằm trên
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlabel('LIME Weight')
    plt.title(f'Sample {count}: {pred_label} (prob={pred_proba[1]:.5f})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load token2id và build id2token
    token2id = json.load(open("./token2id_ransome_mal.json", encoding="utf-8"))
    id2token = build_id2token(token2id)

    # 2. Khởi tạo và build model để load_weights
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("./best_model_ransome_mal.weights.h5")

    # 3. Chuẩn bị dữ liệu SHAP & LIME
    X_background = np.load("./X_background_ransome_mal.npy")
    X_test       = np.load("./X_test_ransome_mal.npy")
    Y_test       = np.load("./Y_test_ransome_mal.npy")

    # 4. Global SHAP: top 10 token theo ảnh hưởng toàn tập
    # global_shap_ransome = shap_explain_global_ransome(cnn_model, X_background, X_test, id2token)
    # print("Global SHAP (top features) decision ransomeware:")
    # for token, value in global_shap_ransome:
    #     print(f"{token}: {value:.6f}")
    # print("-" * 50)
    global_shap_malware = shap_explain_global_malware(cnn_model, X_background, X_test, id2token)
    print("Global SHAP (top features) decision malware:")
    for token, value in global_shap_malware:
        print(f"{token}: {value:.6f}")
    print("-" * 50)

    # 5. Local LIME
    pad_token_id = 0
    for count, X_test_i in enumerate(X_test, 1):
        # Bỏ các token ID là <PAD>
        tokens_without_pad = [x for x in X_test_i if x != pad_token_id]
        if not tokens_without_pad:
            print(f"Sample {count}: Empty after removing PAD, can't predict.")    
            continue
        local_lime = lime_explain_instance(cnn_model, tokens_without_pad, id2token)

        # Dự đoán nhãn cho sample này
        pred_proba = cnn_model.predict(np.array([X_test_i]))[0]
        pred_label = "Ransomware" if pred_proba[1] > 0.5 else "Malware"

        top_feature = local_lime[0][0] if local_lime else None
        if top_feature == '<PAD>':
            print(f"Sample {count}: Top feature is <PAD>, can't predict")

        else:
            print(f"Sample {count}: Local LIME (top features):", local_lime)
            print(f"Sample {count}: {pred_label} from prediction")
        
        if pred_label == "Ransomware":
            if Y_test[count-1]==1:
                print(f"Sample {count}: Correct prediction")
            else:
                print(f"Sample {count}: Incorrect prediction")
            print("-" * 50)
        else:
            if Y_test[count-1]==0:
                print(f"Sample {count}: Correct prediction")
            else:
                print(f"Sample {count}: Incorrect prediction")
            print("-" * 50)
        plot_lime_top_5_5(local_lime, count, pred_label, pred_proba)
