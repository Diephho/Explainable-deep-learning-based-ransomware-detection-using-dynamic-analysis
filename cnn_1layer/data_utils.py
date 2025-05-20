# data_utils.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_LEN_API, MAX_LEN_DLL, MAX_LEN_MUTEX

def build_token_dict(reports):
    """
    Xây từ điển token (API/DLL/Mutex) → ID.
    Padding ID = 0.
    """
    all_tokens = set()
    token2id = {"<PAD>": 0} 
    for r in reports:
        all_tokens.update(r['apis'])
        all_tokens.update(r['dlls'])
        all_tokens.update(r['mutexes'])
    token2id = {tok: i+1 for i, tok in enumerate(sorted(all_tokens))}
    return token2id


def encode_and_pad(seq, token2id, maxlen):
    ids = [token2id.get(tok, 0) for tok in seq]
    return pad_sequences([ids], maxlen=maxlen, padding='post', truncating='post')[0]


def prepare_sequences(reports, token2id):
    """
    Với mỗi báo cáo chứa 'apis','dlls','mutexes',
    trả về mảng shape=(N, SEQ_LEN).
    """
    sequences = []
    for r in reports:
        api_ids   = encode_and_pad(r['apis'],   token2id, MAX_LEN_API)
        dll_ids   = encode_and_pad(r['dlls'],   token2id, MAX_LEN_DLL)
        mutex_ids = encode_and_pad(r['mutexes'], token2id, MAX_LEN_MUTEX)
        seq = np.concatenate([api_ids, dll_ids, mutex_ids], axis=0)
        sequences.append(seq)
    return np.stack(sequences)