# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import dump
from config import RANDOM_STATE, VALIDATION_SPLIT
from data_utils import build_token_dict, prepare_sequences


def load_reports(ransom_dir: str, benign_dir: str):
    reports, labels = [], []
    # Load ransomware reports
    for fname in os.listdir(ransom_dir):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(ransom_dir, fname), 'r', encoding='utf-8') as f:
            reports.append(json.load(f))
        labels.append(1)
    # Load benign reports
    for fname in os.listdir(benign_dir):
        if not fname.lower().endswith('.json'):
            continue
        with open(os.path.join(benign_dir, fname), 'r', encoding='utf-8') as f:
            reports.append(json.load(f))
        labels.append(0)
    return reports, labels


def main():
    # Directories containing JSON feature files
    ransom_dir = 'attributes/ransomware'
    benign_dir = 'attributes/benign'

    # 1) Load data and labels
    reports, labels = load_reports(ransom_dir, benign_dir)

    # 2) Build and save token dictionary
    token2id = build_token_dict(reports)
    with open('token2id.json', 'w', encoding='utf-8') as f:
        json.dump(token2id, f, indent=4)

    # 3) Prepare sequences and labels
    X = prepare_sequences(reports, token2id)
    y = np.array(labels)

    # 4) Split into train+val and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    # 5) Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, stratify=y_temp, random_state=RANDOM_STATE
    )

    # 6) Save for explainers
    np.save('X_background.npy', X_train)
    np.save('X_validate.npy', X_val)
    np.save('Y_validate.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('Y_test.npy', y_test)

    # 7) Train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 8) Save trained model
    dump(clf, 'rf_model.joblib')

    # 9) Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    f1_val  = f1_score(y_val, y_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    tpr_val = tp / (tp + fn)
    fpr_val = fp / (fp + tn)
    print(f"[Validation] Accuracy: {acc_val:.4f}")
    print(f"[Validation] TPR (Recall): {tpr_val:.4f}, FPR: {fpr_val:.4f}, F1-Score: {f1_val:.4f}\n")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, digits=4))

    # 10) Evaluate on test set
    y_test_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test  = f1_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    tpr_test = tp / (tp + fn)
    fpr_test = fp / (fp + tn)
    print(f"[Test] Accuracy: {acc_test:.4f}")
    print(f"[Test] TPR (Recall): {tpr_test:.4f}, FPR: {fpr_test:.4f}, F1-Score: {f1_test:.4f}\n")
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

if __name__ == '__main__':
    main()