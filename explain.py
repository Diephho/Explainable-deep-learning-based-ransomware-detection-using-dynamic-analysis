# explain.py

import numpy as np
import json
from lime.lime_text import LimeTextExplainer
import shap
from data_utils import prepare_sequences
from model import model as build_model
from config import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX, SEQ_LEN

def build_id2token(token2id):
    return {idx: tok for tok, idx in token2id.items()}

def lime_explain_instance(cnn_model, sequence, id2token, num_features=10):
    explainer = LimeTextExplainer(class_names=["benign","ransomware"], split_expression=r"\s+")
    def predict_proba(texts):
        X = np.array([[int(x) for x in t.split()] for t in texts])
        probs = cnn_model.predict(X)
        return np.hstack([1-probs, probs])
    text_input = " ".join(str(int(x)) for x in sequence)
    exp = explainer.explain_instance(text_input, predict_proba, num_features=num_features)
    return [(id2token[int(tok)], weight) for tok, weight in exp.as_list()]

def shap_explain_global(cnn_model, X_background, X_test, id2token):
    try:
        # Dùng GradientExplainer cho TF2.x
        explainer = shap.GradientExplainer(cnn_model, X_background)
        shap_vals = explainer.shap_values(X_test)
    except Exception:
        # Fallback: KernelExplainer (chậm hơn)
        f = lambda x: cnn_model.predict(x)
        explainer = shap.KernelExplainer(f, X_background[:50])
        shap_vals = explainer.shap_values(X_test[:20])
    sv = shap_vals[1]  # cho lớp ransomware

    # Tính mean |SHAP| theo nhóm
    avg_api   = np.abs(sv[:,:MAX_LEN_API]).mean()
    avg_dll   = np.abs(sv[:,MAX_LEN_API:MAX_LEN_API+MAX_LEN_DLL]).mean()
    avg_mutex = np.abs(sv[:,-MAX_LEN_MUTEX:]).mean()

    shap.summary_plot(sv, X_test, feature_names=[id2token[i] for i in range(SEQ_LEN)])
    return {"api_mean_abs_shap": avg_api,
            "dll_mean_abs_shap": avg_dll,
            "mutex_mean_abs_shap": avg_mutex}

if __name__ == "__main__":
    # 1. Load token2id và build id2token
    token2id = json.load(open("token2id.json", encoding="utf-8"))
    id2token = build_id2token(token2id)

    # 2. Khởi tạo và build model để load_weights
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("best_model.weights.h5")

    # 3. Chuẩn bị dữ liệu SHAP & LIME
    X_background = np.load("X_background.npy")
    X_test       = np.load("X_test.npy")

    # 4. Global SHAP
    global_shap = shap_explain_global(cnn_model, X_background, X_test, id2token)
    print("Global SHAP:", global_shap)

    # 5. Local LIME
    local_lime = lime_explain_instance(cnn_model, X_test[0], id2token)
    print("Local LIME (top features):", local_lime)
