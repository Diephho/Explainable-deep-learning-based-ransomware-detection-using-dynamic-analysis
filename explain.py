# explain.py

import numpy as np
import json
from lime.lime_text import LimeTextExplainer
import shap
from data_utils import prepare_sequences
from model import model as build_model
from config import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX

def build_id2token(token2id):
    """Tạo dict id->token từ token2id."""
    return {idx: tok for tok, idx in token2id.items()}

def lime_explain_instance(cnn_model, sequence, token2id, id2token, num_features=10):
    """
    Giải thích cục bộ cho 1 sample sequence bằng LIME.
    - sequence: np.array shape (SEQ_LEN,)
    - token2id, id2token: ánh xạ giữa token/string và ID
    Trả về list [(token, weight), ...].
    """
    explainer = LimeTextExplainer(
        class_names=["benign", "ransomware"],
        split_expression=r"\s+"
    )

    def predict_proba(texts):
        # texts: list of strings dạng "12 45 3 0 78 ..."
        seqs = []
        for t in texts:
            ids = [int(x) for x in t.split()]
            seqs.append(ids)
        X = np.array(seqs)
        probs = cnn_model.predict(X)
        # Trả về mảng [[p0,p1], ...]
        return np.hstack([1 - probs, probs])

    # Chuyển sequence thành string
    text_input = " ".join(str(int(x)) for x in sequence)
    exp = explainer.explain_instance(text_input, predict_proba, num_features=num_features)

    # Chuyển token_id sang token string
    return [(id2token[int(tok)], weight) for tok, weight in exp.as_list()]

def shap_explain_global(cnn_model, X_background, X_test, id2token):
    """
    Giải thích toàn cục với SHAP.
    - X_background: np.array shape (B, SEQ_LEN)
    - X_test:        np.array shape (N, SEQ_LEN)
    Trả về dict trung bình |SHAP| cho nhóm API/DLL/Mutex.
    """
    # DeepExplainer cho Keras model
    explainer = shap.DeepExplainer(cnn_model, X_background)
    shap_vals = explainer.shap_values(X_test)
    # shap_vals là list 2 mảng: shap_vals[0] cho lớp benign, [1] cho ransomware
    sv = shap_vals[1]

    # Tính trung bình |shap| theo nhóm
    avg_api   = np.abs(sv[:, :MAX_LEN_API]).mean()
    avg_dll   = np.abs(sv[:, MAX_LEN_API:MAX_LEN_API+MAX_LEN_DLL]).mean()
    avg_mutex = np.abs(sv[:, -MAX_LEN_MUTEX:]).mean()

    # Vẽ summary plot (tùy chọn)
    shap.summary_plot(
        sv, 
        X_test, 
        feature_names=[id2token[i] for i in range(X_test.shape[1])]
    )

    return {
        "api_mean_abs_shap": avg_api,
        "dll_mean_abs_shap": avg_dll,
        "mutex_mean_abs_shap": avg_mutex
    }

if __name__ == "__main__":
    # Ví dụ khởi tạo và sử dụng:
    # 1. Load hoặc train model
    token2id = json.load(open("token2id.json"))  # ví dụ lưu sẵn
    id2token = build_id2token(token2id)

    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.load_weights("best_model.weights.h5")       # nếu bạn đã lưu weights

    # 2. Chuẩn bị data cho SHAP
    X_background = np.load("X_background.npy")   # ví dụ phần nền
    X_test       = np.load("X_test.npy")         # ví dụ batch cần giải thích

    # 3. Chạy SHAP global
    global_shap = shap_explain_global(cnn_model, X_background, X_test, id2token)
    print("Global SHAP:", global_shap)

    # 4. Chạy LIME cho một sample
    sample_seq = X_test[0]
    local_lime = lime_explain_instance(cnn_model, sample_seq, token2id, id2token)
    print("Local LIME (top features):", local_lime)
