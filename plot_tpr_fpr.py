# plot_tpr_fpr.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_tpr_fpr.py

1) Grid-search trên API=[50,100,500,1000], DLL=[5,10,15], Mutex=[5,10,15]
2) Tính TPR và FPR (FPR = FP/(FP+TN)) cho mỗi cấu hình.
3) Vẽ 3 mặt biên (base/front/right) của khối lập phương, ticks bằng giá trị thử nghiệm.
"""
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import config
from data_utils import build_token_dict, encode_and_pad
from model import model as build_model
import json

def load_reports(ransom_dir, benign_dir):
    reports, labels = [], []
    for folder, lab in [(ransom_dir,1),(benign_dir,0)]:
        for fn in os.listdir(folder):
            if fn.lower().endswith('.json'):
                with open(os.path.join(folder, fn), 'r', encoding='utf-8') as f:
                    reports.append(json.load(f))
                labels.append(lab)
    return reports, np.array(labels)


def run_experiment(reports, labels, token2id, api_len, dll_len, mutex_len):
    # update config lengths
    config.MAX_LEN_API   = api_len
    config.MAX_LEN_DLL   = dll_len
    config.MAX_LEN_MUTEX = mutex_len
    config.SEQ_LEN       = api_len + dll_len + mutex_len

    # encode + pad
    X = []
    for r in reports:
        a = encode_and_pad(r['apis'],    token2id, api_len)
        d = encode_and_pad(r['dlls'],    token2id, dll_len)
        m = encode_and_pad(r['mutexes'], token2id, mutex_len)
        X.append(np.concatenate([a,d,m]))
    X = np.stack(X)
    y = labels

    # split (random each run)
    seed = np.random.randint(0, 10000)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=seed
    )

    # train
    cnn = build_model(vocab_size=len(token2id)+1)
    cnn.fit(
        Xtr, ytr,
        validation_split=config.VALIDATION_SPLIT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=0
    )

    # eval
    ypr = np.argmax(cnn.predict(Xte), axis=1)
    tn, fp, fn, tp = confusion_matrix(yte, ypr).ravel()
    tpr = tp/(tp+fn) if tp+fn>0 else 0.0
    fpr = fp/(fp+tn) if fp+tn>0 else 0.0
    return tpr, fpr


def plot_three_faces(ax, api_vals, dll_vals, mut_vals, data, cmap):
    """
    Vẽ 3 mặt biên của cube:
      - Base (Z = mut_min) trên mặt API×DLL
      - Front (Y = dll_min) trên mặt API×Mutex
      - Right (X = api_max) trên mặt DLL×Mutex
    """
    norm = plt.Normalize(data.min(), data.max())
    cmap_fn = plt.cm.get_cmap(cmap)

    # Base face: Z = mut_vals[0]
    A, D = np.meshgrid(api_vals, dll_vals, indexing='ij')
    Z0 = np.full_like(A, mut_vals[0], dtype=float)
    V0 = data[:,:,0]
    ax.plot_surface(
        A, D, Z0,
        facecolors=cmap_fn(norm(V0)),
        rstride=1, cstride=1, shade=False
    )

    # Front face: Y = dll_vals[0]
    A2, M2 = np.meshgrid(api_vals, mut_vals, indexing='ij')
    Y0 = dll_vals[0]
    V1 = data[:,0,:]
    ax.plot_surface(
        A2, np.full_like(A2, Y0), M2,
        facecolors=cmap_fn(norm(V1)),
        rstride=1, cstride=1, shade=False
    )

    # Right face: X = api_vals[-1]
    D3, M3 = np.meshgrid(dll_vals, mut_vals, indexing='ij')
    X1 = api_vals[-1]
    V2 = data[-1,:,:]
    ax.plot_surface(
        np.full_like(D3, X1), D3, M3,
        facecolors=cmap_fn(norm(V2)),
        rstride=1, cstride=1, shade=False
    )

    # set ticks exactly at values
    ax.set_xticks(api_vals)
    ax.set_yticks(dll_vals)
    ax.set_zticks(mut_vals)
    ax.set_xlabel('API Call Seq Length')
    ax.set_ylabel('DLL Seq Length')
    ax.set_zlabel('Mutex Seq Length')

    # equal aspect for cube
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=30, azim=-60)

if __name__ == '__main__':
    api_vals = [50, 100, 500, 1000]
    dll_vals = [5, 10, 15]
    mut_vals = [5, 10, 15]

    reports, labels = load_reports(
        'attributes/ransomware',
        'attributes/benign'
    )
    token2id = build_token_dict(reports)

    grid = {}
    for a,d,m in itertools.product(api_vals, dll_vals, mut_vals):
        print(f'API={a}, DLL={d}, MUTEX={m}...', end=' ')
        tpr, fpr = run_experiment(reports, labels, token2id, a, d, m)
        print(f'TPR={tpr:.5f}, FPR={fpr:.5f}')
        grid[(a,d,m)] = {'tpr':tpr, 'fpr':fpr}

    shape = (len(api_vals), len(dll_vals), len(mut_vals))
    TPR = np.zeros(shape)
    FPR = np.zeros(shape)
    for i,a in enumerate(api_vals):
        for j,d in enumerate(dll_vals):
            for k,m in enumerate(mut_vals):
                TPR[i,j,k] = grid[(a,d,m)]['tpr']
                FPR[i,j,k] = grid[(a,d,m)]['fpr']

    for data, cmap, title, fname in [
        (TPR, 'Greens', '(a) TPR plot', 'tpr_cube.png'),
        (FPR, 'Reds',   '(b) FPR plot', 'fpr_cube.png')
    ]:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        plot_three_faces(ax, api_vals, dll_vals, mut_vals, data, cmap)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(data.min(), data.max()))
        fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label=title)
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.show()
