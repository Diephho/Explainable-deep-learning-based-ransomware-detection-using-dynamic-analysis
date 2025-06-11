# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config_ransome_mal import RANDOM_STATE, VALIDATION_SPLIT, BATCH_SIZE, EPOCHS
from data_utils import build_token_dict, prepare_sequences
from model import model
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def load_reports(ransom_dir: str, mal_dir: str):
    reports, labels, file_names = [], [], []
    for fname in os.listdir(ransom_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(ransom_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(1)
        file_names.append(fname)
    for fname in os.listdir(mal_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(mal_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(0)
        file_names.append(fname)
    return reports, labels, file_names

def main():
    ransom_dir = 'attributes/ransomware'
    mal_dir = 'attributes/malware'
    reports, labels, file_names = load_reports(ransom_dir, mal_dir)

    token2id  = build_token_dict(reports)
    with open('token2id_ransome_mal.json', 'w', encoding='utf-8') as f:
        json.dump(token2id, f, indent=4)

    sequences = prepare_sequences(reports, token2id)
    X = sequences
    y = np.array(labels)  # ✅ KHÔNG one-hot

    X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
        X, y, file_names,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_background = X_train
    np.save('X_background_ransome_mal.npy', X_background)
    np.save('X_test_ransome_mal.npy', X_test)
    np.save('Y_test_ransome_mal.npy', y_test)
    np.save('file_names_test_ransome_mal.npy', fn_test)

    X_train, X_validate, y_train, y_validate, fn_train, fn_validate = train_test_split(
        X, y, file_names,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_background_validate = X_validate
    np.save('X_background_validate_ransome_mal.npy', X_background_validate)
    np.save('X_validate_ransome_mal.npy', X_validate)
    np.save('Y_validate_ransome_mal.npy', y_validate)
    np.save('file_names_validate_ransome_mal.npy', fn_validate)

    cnn_model = model(vocab_size=len(token2id) + 1)
    cnn_model.summary()

    history = cnn_model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    cnn_model.save_weights('best_model_ransome_mal.weights.h5')

    loss, acc = cnn_model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    print(f"{'-'*50}")
    y_pred_probs = cnn_model.predict(X_test)
    y_pred_labels = (y_pred_probs[:, 1] > 0.5).astype(int)  # Chỉ số 1 là "Ransomware"
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels).ravel()
    accuracy = accuracy_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)       # TPR
    f1 = f1_score(y_test, y_pred_labels)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0        # FPR = False Positive Rate
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Recall (TPR): {recall:.4f}")
    print(f"Test FPR      : {fpr:.4f}")
    print(f"Test F1-Score : {f1:.4f}")

if __name__ == '__main__':
    main()
