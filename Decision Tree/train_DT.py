# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from joblib import dump
from config import RANDOM_STATE, VALIDATION_SPLIT
from data_utils import build_token_dict, prepare_sequences
from model_DT import model

def load_reports(ransom_dir: str, benign_dir: str):
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
    # Load benign reports
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
    # Directories containing JSON feature files
    ransom_dir = 'attributes/ransomware'
    benign_dir = 'attributes/benign'

    # Load data
    reports, labels = load_reports(ransom_dir, benign_dir)

    # Build and save token dictionary
    token2id = build_token_dict(reports)
    with open('token2id_DT.json', 'w', encoding='utf-8') as f:
        json.dump(token2id, f, indent=4)

    # Prepare input sequences
    X = prepare_sequences(reports, token2id)
    y = np.array(labels)

    # Split into temporary set and final test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # Split temporary into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDATION_SPLIT,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Save datasets for explainers
    np.save('X_background_DT.npy', X_train)
    np.save('X_validate_DT.npy', X_val)
    np.save('Y_validate_DT.npy', y_val)
    np.save('X_test_DT.npy', X_test)
    np.save('Y_test_DT.npy', y_test)

    # Initialize and train Decision Tree model
    clf = model()
    clf.fit(X_train, y_train)

    # Save trained model
    dump(clf, 'dt_model.joblib')

    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    tpr_val = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr_val = fp / (fp + tn)  # False Positive Rate
    print("[Validation] Accuracy: {:.4f}".format(acc_val))
    print("[Validation] TPR (Recall): {:.4f}, FPR: {:.4f}, F1-Score: {:.4f}".format(tpr_val, fpr_val, f1_val))
    print(classification_report(y_val, y_val_pred, digits=4))

    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    tpr_test = tp / (tp + fn)
    fpr_test = fp / (fp + tn)
    print("[Test] Accuracy: {:.4f}".format(acc_test))
    print("[Test] TPR (Recall): {:.4f}, FPR: {:.4f}, F1-Score: {:.4f}".format(tpr_test, fpr_test, f1_test))
    print(classification_report(y_test, y_test_pred, digits=4))

if __name__ == '__main__':
    main()
