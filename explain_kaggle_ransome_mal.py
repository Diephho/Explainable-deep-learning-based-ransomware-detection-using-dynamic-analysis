# explain.py

import numpy as np
import json
from lime.lime_text import LimeTextExplainer
import shap
from data_utils import prepare_sequences
from model import model as build_model
from config_ransome_mal import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX, SEQ_LEN
from sklearn.metrics import classification_report, confusion_matrix

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
    X_display = X_test[:sv.shape[0]]

    # Tính mean |SHAP| theo nhóm
    avg_api   = np.abs(sv[:,:MAX_LEN_API]).mean()
    avg_dll   = np.abs(sv[:,MAX_LEN_API:MAX_LEN_API+MAX_LEN_DLL]).mean()
    avg_mutex = np.abs(sv[:,-MAX_LEN_MUTEX:]).mean()

    shap.summary_plot(sv, X_display, feature_names=[id2token.get(i, f"<UNK_{i}>") for i in range(SEQ_LEN)])
    return {"api_mean_abs_shap": avg_api,
            "dll_mean_abs_shap": avg_dll,
            "mutex_mean_abs_shap": avg_mutex}

if __name__ == "__main__":
    # 1. Load token2id và build id2token
    token2id = json.load(open("/kaggle/input/xran-demo/token2id_ransome_mal.json", encoding="utf-8"))
    id2token = build_id2token(token2id)

    # 2. Khởi tạo và build model để load_weights
    cnn_model = build_model(vocab_size=len(token2id)+1)
    cnn_model.build((None, SEQ_LEN))
    cnn_model.load_weights("/kaggle/input/xran-demo/best_model_ransome_mal.weights.h5")

    # 3. Chuẩn bị dữ liệu SHAP & LIME
    X_background = np.load("/kaggle/input/xran-demo/X_background_ransome_mal.npy")
    X_test       = np.load("/kaggle/input/xran-demo/X_test_ransome_mal.npy")
    Y_test       = np.load("/kaggle/input/xran-demo/Y_test_ransome_mal.npy")

    # 4. Global SHAP
    global_shap = shap_explain_global(cnn_model, X_background, X_test, id2token)
    print("Global SHAP:", global_shap)

    # 5. LIME for Each Sample
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
            print(f"Sample {count}: Top feature is <PAD>, can't predict.")

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

    # Predict toàn bộ
    y_pred_probs = cnn_model.predict(X_test)
    y_pred = (y_pred_probs[:, 1] > 0.5).astype(int)  # lấy xác suất ransomware giống như pred_label

    # In báo cáo
    print("[Classification Report on Test Set]")
    print(classification_report(Y_test, y_pred, digits=4, target_names=["Malware", "Ransomware"]))

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    tpr = tp / (tp + fn)  # Sensitivity, Recall
    fpr = fp / (fp + tn)  # Fall-out

    print(f"TPR (Recall): {tpr:.4f}")
    print(f"FPR: {fpr:.4f}")
    