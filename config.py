# Chuỗi đầu vào: API (500) + DLL (10) + Mutex (10)
MAX_LEN_API = 500
MAX_LEN_DLL = 10
MAX_LEN_MUTEX = 10
SEQ_LEN = MAX_LEN_API + MAX_LEN_DLL + MAX_LEN_MUTEX

# Embedding dimension
EMB_DIM = 128

# Huấn luyện
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Decision Tree hyper-parameters
DT_MAX_DEPTH    = None    # hoặc số nguyên, ví dụ 10
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF  = 1
DT_CRITERION        = "gini"  # hoặc "entropy"