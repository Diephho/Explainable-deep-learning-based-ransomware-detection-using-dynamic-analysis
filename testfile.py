import numpy as np

# Đổi tên file dưới đây cho đúng với file bạn đã lưu
file_names = np.load("file_names_test.npy", allow_pickle=True)

for name in file_names:
    print(name)