# explain.py

import numpy as np
import json
from lime.lime_text import LimeTextExplainer
import shap
from data_utils import prepare_sequences
from model import model as build_model
from config import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX, SEQ_LEN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

def build_id2token(token2id):
    id2token = {int(idx): tok for tok, idx in token2id.items()}
    if 0 not in id2token:
        id2token[0] = "<PAD>"
    return id2token


def lime_explain_instance(cnn_model, sequence, id2token, num_features=20):
    explainer = LimeTextExplainer(class_names=["benign","ransomware"], split_expression=r"\s+")
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
        return probs
    text_input = " ".join(str(int(x)) for x in sequence)
    exp = explainer.explain_instance(text_input, predict_proba, num_features=num_features)
    return [(id2token[int(tok)], weight) for tok, weight in exp.as_list()]

def plot_top_shap_bar(top_tokens, type):
    features, weights = zip(*top_tokens)
    colors = ['red' if w > 0 else 'blue' for w in weights]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, weights, color=colors, edgecolor='black')

    # Vẽ đường x = 0
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # Chú giải
    red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Pushes to Ransomware')
    blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Pushes to Benign')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')

    # Căn chỉnh
    plt.xlabel('Mean SHAP Value', fontsize=12)
    plt.title(f'Global SHAP Values for {type} Prediction', fontsize=14)
    plt.gca().invert_yaxis()  # Đảo ngược trục y để giá trị lớn nằm trên
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def shap_explain_global_ransome(cnn_model, X_background, X_test, id2token, top_n=10):
    try:
        explainer = shap.GradientExplainer(cnn_model, X_background)
        shap_vals = explainer.shap_values(X_test)
    except Exception:
        f = lambda x: cnn_model.predict(x)
        explainer = shap.KernelExplainer(f, X_background[:50])
        shap_vals = explainer.shap_values(X_test[:20])

    # SHAP values cho lớp ransomware (index 1)
    sv = shap_vals[1]  # shape: (num_samples, seq_len)
    X_display = X_test[:sv.shape[0]]  # cùng số sample

    # Mean SHAP giữ nguyên dấu
    mean_shap = sv.mean(axis=0)

    # Lấy token ID theo từng vị trí trung bình và ánh xạ sang token
    token_ids = X_display[:, 0:sv.shape[1]].mean(axis=0).astype(int)
    feature_names = [id2token.get(i, f"Pos_{idx}") for idx, i in enumerate(token_ids)]

    # Ghép tên + SHAP value
    token_shap_pairs = list(zip(feature_names, mean_shap))

    # Top đặc trưng đẩy về ransomware
    top_pos = sorted(token_shap_pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # Top đặc trưng đẩy về benign
    top_neg = sorted(token_shap_pairs, key=lambda x: x[1])[:top_n]

    # Vẽ tổng quan SHAP
    shap.summary_plot(sv, X_display, feature_names=feature_names)

    # Bar plot top tích cực và tiêu cực (nếu bạn định vẽ luôn)
    plot_top_shap_bar(top_pos + top_neg,'ransomware')  # hoặc tách riêng nếu muốn

    return top_pos+top_neg

def shap_explain_global_benign(cnn_model, X_background, X_test, id2token, top_n=10):
    try:
        explainer = shap.GradientExplainer(cnn_model, X_background)
        shap_vals = explainer.shap_values(X_test)
    except Exception:
        f = lambda x: cnn_model.predict(x)
        explainer = shap.KernelExplainer(f, X_background[:50])
        shap_vals = explainer.shap_values(X_test[:20])

    # SHAP values cho lớp benign (index 0)
    sv = shap_vals[0]  # shape: (num_samples, seq_len)
    X_display = X_test[:sv.shape[0]]  # cùng số sample

    # Mean SHAP giữ nguyên dấu
    mean_shap = sv.mean(axis=0)

    # Lấy token ID theo từng vị trí trung bình và ánh xạ sang token
    token_ids = X_display[:, 0:sv.shape[1]].mean(axis=0).astype(int)
    feature_names = [id2token.get(i, f"Pos_{idx}") for idx, i in enumerate(token_ids)]

    # Ghép tên + SHAP value
    token_shap_pairs = list(zip(feature_names, mean_shap))

    # Top đặc trưng đẩy về ransomware
    top_pos = sorted(token_shap_pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # Top đặc trưng đẩy về benign
    top_neg = sorted(token_shap_pairs, key=lambda x: x[1])[:top_n]

    # Vẽ tổng quan SHAP
    shap.summary_plot(sv, X_display, feature_names=feature_names)

    # Bar plot top tích cực và tiêu cực (nếu bạn định vẽ luôn)
    plot_top_shap_bar(top_pos + top_neg, 'benign')  # hoặc tách riêng nếu muốn

    return top_pos+top_neg

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
    token2id = json.load(open("./token2id.json", encoding="utf-8"))
    id2token = build_id2token(token2id)

    # 2. Khởi tạo và build model để load_weights
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("./best_model.weights.h5")

    # 3. Chuẩn bị dữ liệu SHAP & LIME
    X_background = np.load("./X_background.npy")
    X_test       = np.load("./X_test.npy")
    Y_test       = np.load("./Y_test.npy")

    # # 4. Global SHAP: top 10 token theo ảnh hưởng toàn tập
    # global_shap_ransome = shap_explain_global_ransome(cnn_model, X_background, X_test, id2token)
    # print("Global SHAP (top features) decision ransomeware:")
    # for token, value in global_shap_ransome:
    #     print(f"{token}: {value:.6f}")
    print("-" * 50)
    global_shap_benign = shap_explain_global_benign(cnn_model, X_background, X_test, id2token)
    print("Global SHAP (top features) decision benign:")
    for token, value in global_shap_benign:
        print(f"{token}: {value:.6f}")
    print("-" * 50)

    # 5. Local LIME
    pad_token_id = 0 
    for count, X_test_i in enumerate(X_test, 1):
        # Bỏ các token ID là <PAD>
        tokens_without_pad = [x for x in X_test_i if x != pad_token_id]
        if not tokens_without_pad:
            print(f"Sample {count}: Empty after removing PAD, likely benign file.")    
            if Y_test[count-1]==0:
                print(f"Sample {count}: Correct prediction")
            else:
                print(f"Sample {count}: Incorrect prediction")
            print("-" * 50)
            continue
        local_lime = lime_explain_instance(cnn_model, tokens_without_pad, id2token)

        # Dự đoán nhãn cho sample này
        pred_proba = cnn_model.predict(np.array([X_test_i]))[0]
        pred_label = "Ransomware" if pred_proba[1] > 0.5 else "Benign"

        top_feature = local_lime[0][0] if local_lime else None
        if top_feature == '<PAD>':
            print(f"Sample {count}: Top feature is <PAD>, likely benign file.")
            print(f"Sample {count} is {pred_label} from DeepLearning")

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
        # Plot horizontal bar chart
        plot_lime_top_5_5(local_lime, count, pred_label, pred_proba)

