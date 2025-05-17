# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, VALIDATION_SPLIT, BATCH_SIZE, EPOCHS
from data_utils import build_token_dict, prepare_sequences
from model import model

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
    ransom_dir = 'attributes/ransomware'
    benign_dir = 'attributes/benign'
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

    background_size = min(100, X_train.shape[0])
    X_background = X_train[:background_size]
    np.save('X_background.npy', X_background)
    np.save('X_test.npy', X_test)

    cnn_model = model(vocab_size=len(token2id) + 1)
    cnn_model.summary()

    history = cnn_model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    cnn_model.save_weights('best_model.weights.h5')

    loss, acc = cnn_model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
