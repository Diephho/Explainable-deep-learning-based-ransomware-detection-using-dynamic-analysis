#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from statistics import mean

def gather_counts(dir_path):
    """
    Duyệt qua tất cả file .json trong dir_path,
    đọc các trường 'dlls','apis','mutexes' và trả về
    3 list chứa số lượng tương ứng trên mỗi file.
    """
    dll_counts   = []
    api_counts   = []
    mutex_counts = []
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith('.json'):
            continue
        full = os.path.join(dir_path, fname)
        try:
            with open(full, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dll_counts.append(   len(data.get('dlls',   [])) )
            api_counts.append(   len(data.get('apis',   [])) )
            mutex_counts.append( len(data.get('mutexes',[])) )
        except Exception as e:
            print(f"⚠️ Lỗi đọc {full}: {e}")
    return dll_counts, api_counts, mutex_counts

def print_stats(name, dlls, apis, mutexes):
    """
    In kết quả trung bình và số file.
    """
    n = len(dlls)
    print(f"\n--- Thống kê cho '{name}' ---")
    if n == 0:
        print("Không có file để tính toán.")
        return
    print(f"Avg. DLLs   : {mean(dlls):.2f}")
    print(f"Avg. APIs   : {mean(apis):.2f}")
    print(f"Avg. Mutexes: {mean(mutexes):.2f}")

if __name__ == "__main__":
    base = "attributes"
    benign_dir   = os.path.join(base, "benign")
    ransom_dir   = os.path.join(base, "ransomware")

    # 1) Thống kê riêng từng thư mục
    dll_b, api_b, mut_b = gather_counts(benign_dir)
    dll_r, api_r, mut_r = gather_counts(ransom_dir)

    print_stats("benign",    dll_b, api_b, mut_b)
    print_stats("ransomware",dll_r, api_r, mut_r)

    # 2) Tổng hợp toàn bộ
    dll_all = dll_b + dll_r
    api_all = api_b + api_r
    mut_all = mut_b + mut_r
    print_stats("attributes (tổng)", dll_all, api_all, mut_all)
