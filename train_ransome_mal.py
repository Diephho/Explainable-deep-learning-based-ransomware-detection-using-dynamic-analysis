# train.py
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config_ransome_mal import RANDOM_STATE, VALIDATION_SPLIT, BATCH_SIZE, EPOCHS
from data_utils import build_token_dict, prepare_sequences
from model import model

def load_reports(ransom_dir: str, mal_dir: str):
    reports, labels = [], []
    for fname in os.listdir(ransom_dir):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(ransom_dir, fname)
        with open(full, 'r', encoding='utf-8') as f:
            features = json.load(f)
        reports.append(features)
        labels.append(1)
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
    ransom_dir = 'attributes/ransomware'
    mal_dir = 'attributes/malware'
    reports, labels = load_reports(ransom_dir, mal_dir)

    token2id  = build_token_dict(reports)
    with open('token2id_ransome_mal.json', 'w', encoding='utf-8') as f:
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
    np.save('X_background_ransome_mal.npy', X_background)
    np.save('X_test_ransome_mal.npy', X_test)
    np.save('Y_test_ransome_mal.npy', y_test)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_background_validate = X_validate
    np.save('X_background_validate_ransome_mal.npy', X_background_validate)
    np.save('X_validate_ransome_mal.npy', X_validate)
    np.save('Y_validate_ransome_mal.npy', y_validate)


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

if __name__ == '__main__':
    main()
