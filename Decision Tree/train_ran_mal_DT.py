# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from joblib import dump
from config_ransome_mal import RANDOM_STATE, VALIDATION_SPLIT
from data_utils import build_token_dict, prepare_sequences


def load_reports(ransom_dir: str, mal_dir: str):
    reports, labels = [], []
    # Load ransomware reports
    for fname in os.listdir(ransom_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(ransom_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(1)
    # Load generic malware reports
    for fname in os.listdir(mal_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(mal_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(0)
    return reports, labels


def main():
    # Paths
    ransom_dir = 'attributes/ransomware'
    mal_dir    = 'attributes/malware'

    # 1) Load and label data
    reports, labels = load_reports(ransom_dir, mal_dir)

    # 2) Build token dictionary
    token2id = build_token_dict(reports)
    with open('token2id_ransome_mal.json', 'w', encoding='utf-8') as f:
        json.dump(token2id, f, indent=4)

    # 3) Prepare sequences
    X = prepare_sequences(reports, token2id)
    y = np.array(labels)

    # 4) Split into train_val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # 5) Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDATION_SPLIT,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # 6) Save datasets for explainers
    np.save('X_background_ransome_mal.npy', X_train)
    np.save('X_validate_ransome_mal.npy', X_val)
    np.save('Y_validate_ransome_mal.npy', y_val)
    np.save('X_test_ransome_mal.npy', X_test)
    np.save('Y_test_ransome_mal.npy', y_test)

    # 7) Train Decision Tree
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # 8) Save model
    dump(clf, 'dt_ransome_mal_model.joblib')

    # 9) Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr = fp / (fp + tn)  # False Positive Rate

    print(f"Test Accuracy   : {acc:.4f}")
    print(f"Test TPR (Recall): {tpr:.4f}")
    print(f"Test FPR        : {fpr:.4f}")
    print(f"Test F1-Score   : {f1:.4f}\n")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == '__main__':
    main()