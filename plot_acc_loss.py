#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_acc_loss.py

1) Grid-search trên API=[50,100,500,1000], DLL=[5,10,15], Mutex=[5,10,15]
2) Tính Test accuracy và loss với Keras evaluate().
3) Vẽ 3 mặt biên của cube (base/front/right), ticks đúng bằng giá trị thử nghiệm.
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import json

import config
from data_utils import build_token_dict, encode_and_pad
from model import model as build_model

def load_reports(ransom_dir, benign_dir):
    reports, labels = [], []
    for folder, lab in [(ransom_dir,1),(benign_dir,0)]:
        for fn in os.listdir(folder):
            if fn.lower().endswith('.json'):
                with open(os.path.join(folder,fn),'r',encoding='utf-8') as f:
                    reports.append(json.load(f))
                labels.append(lab)
    return reports, np.array(labels)

def run_experiment(reports, labels, token2id, api_len, dll_len, mutex_len):
    # update config lengths
    config.MAX_LEN_API   = api_len
    config.MAX_LEN_DLL   = dll_len
    config.MAX_LEN_MUTEX = mutex_len
    config.SEQ_LEN       = api_len + dll_len + mutex_len

    # encode & pad
    X = []
    for r in reports:
        a = encode_and_pad(r['apis'],    token2id, api_len)
        d = encode_and_pad(r['dlls'],    token2id, dll_len)
        m = encode_and_pad(r['mutexes'], token2id, mutex_len)
        X.append(np.concatenate([a,d,m]))
    X = np.stack(X)
    y = labels

    # split
    seed = np.random.randint(0, 10000)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=seed
    )

    # build & train
    cnn = build_model(vocab_size=len(token2id)+1)
    cnn.fit(
        Xtr, ytr,
        validation_split=config.VALIDATION_SPLIT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=0
    )

    # evaluate
    loss, acc = cnn.evaluate(Xte, yte, verbose=0)
    return acc, loss

def plot_three_faces(ax, api_vals, dll_vals, mut_vals, data, cmap):
    norm = plt.Normalize(data.min(), data.max())
    cmap_fn = plt.cm.get_cmap(cmap)

    # 1) Base face, Z = mut_vals[0]
    A, D = np.meshgrid(api_vals, dll_vals, indexing='ij')
    Z0 = np.full_like(A, mut_vals[0], dtype=float)
    V0 = data[:, :, 0]
    ax.plot_surface(
        A, D, Z0,
        facecolors=cmap_fn(norm(V0)),
        rstride=1, cstride=1, shade=False
    )

    # 2) Front face, Y = dll_vals[0]
    A2, M2 = np.meshgrid(api_vals, mut_vals, indexing='ij')
    Y0 = dll_vals[0]
    V1 = data[:, 0, :]
    ax.plot_surface(
        A2, np.full_like(A2, Y0), M2,
        facecolors=cmap_fn(norm(V1)),
        rstride=1, cstride=1, shade=False
    )

    # 3) Right face, X = api_vals[-1]
    D3, M3 = np.meshgrid(dll_vals, mut_vals, indexing='ij')
    X1 = api_vals[-1]
    V2 = data[-1, :, :]
    ax.plot_surface(
        np.full_like(D3, X1), D3, M3,
        facecolors=cmap_fn(norm(V2)),
        rstride=1, cstride=1, shade=False
    )

    # ticks & labels
    ax.set_xticks(api_vals)
    ax.set_yticks(dll_vals)
    ax.set_zticks(mut_vals)
    ax.set_xlabel('API Call Seq Length')
    ax.set_ylabel('DLL Seq Length')
    ax.set_zlabel('Mutex Seq Length')

    # ép cube đều, góc nhìn cho dễ quan sát
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=30, azim=-60)

if __name__ == '__main__':
    # 1) Thiết lập dải khảo sát
    api_vals = [50, 100, 500, 1000]
    dll_vals = [5, 10, 15]
    mut_vals = [5, 10, 15]

    # 2) Load data
    reports, labels = load_reports(
        'attributes/ransomware',
        'attributes/benign'
    )
    token2id = build_token_dict(reports)

    # 3) Chạy grid-search
    grid = {}
    for a,d,m in itertools.product(api_vals, dll_vals, mut_vals):
        print(f'API={a}, DLL={d}, MUTEX={m}...', end=' ')
        acc, loss = run_experiment(reports, labels, token2id, a,d,m)
        print(f'Acc={acc:.5f}, Loss={loss:.5f}')
        grid[(a,d,m)] = {'acc':acc, 'loss':loss}

    # pack vào arrays
    shape = (len(api_vals), len(dll_vals), len(mut_vals))
    ACC  = np.zeros(shape)
    LOSS = np.zeros(shape)
    for i,a in enumerate(api_vals):
      for j,d in enumerate(dll_vals):
        for k,m in enumerate(mut_vals):
          ACC[i,j,k]  = grid[(a,d,m)]['acc']
          LOSS[i,j,k] = grid[(a,d,m)]['loss']

    # 4) Vẽ và lưu
    for data, cmap, title, fname in [
        (ACC,  'Blues',   '(a) Accuracy plot', 'acc_cube.png'),
        (LOSS, 'Purples', '(b) Loss plot',     'loss_cube.png')
    ]:
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111, projection='3d')
        plot_three_faces(ax, api_vals, dll_vals, mut_vals, data, cmap)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(data.min(),data.max()),
                                  cmap=cmap),
            ax=ax, shrink=0.6, aspect=10,
            label=title
        )
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.show()
