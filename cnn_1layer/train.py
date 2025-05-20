# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, VALIDATION_SPLIT, BATCH_SIZE, EPOCHS
from data_utils import build_token_dict, prepare_sequences
from model import model
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def load_reports(ransom_dir: str, benign_dir: str):
    reports, labels = [], []
    for fname in os.listdir(ransom_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(ransom_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(1)
    for fname in os.listdir(benign_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(benign_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(0)
    return reports, labels

def main():
    ransom_dir = '../attributes/ransomware'
    benign_dir = '../attributes/benign'
    reports, labels = load_reports(ransom_dir, benign_dir)

    token2id  = build_token_dict(reports)
    with open('token2id.json', 'w', encoding='utf-8') as f:
        json.dump(token2id, f, indent=4)

    sequences = prepare_sequences(reports, token2id)
    X = sequences
    y = np.array(labels)  # ✅ KHÔNG one-hot

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_background = X_train
    np.save('X_background.npy', X_background)
    np.save('X_test.npy', X_test)
    np.save('Y_test.npy', y_test)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_background_validate = X_validate
    np.save('X_background_validate.npy', X_background_validate)
    np.save('X_validate.npy', X_validate)
    np.save('Y_validate.npy', y_validate)


    cnn_model = model(vocab_size=len(token2id) + 1)
    cnn_model.summary()

    history = cnn_model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    cnn_model.save_weights('best_model.weights.h5')

    loss, acc = cnn_model.evaluate(X_validate, y_validate)
    print(f"Validate loss: {loss:.4f}, Validate accuracy: {acc:.4f}")

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
