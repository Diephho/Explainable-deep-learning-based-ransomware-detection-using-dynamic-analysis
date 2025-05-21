# model.py
from sklearn.tree import DecisionTreeClassifier
from config import (
    RANDOM_STATE,
    DT_MAX_DEPTH,
    DT_MIN_SAMPLES_SPLIT,
    DT_MIN_SAMPLES_LEAF,
    DT_CRITERION
)

def model() -> DecisionTreeClassifier:
    """
    Trả về một DecisionTreeClassifier đã config sẵn.
    Input X: array shape (n_samples, SEQ_LEN)
    """
    dt = DecisionTreeClassifier(
        criterion=DT_CRITERION,
        max_depth=DT_MAX_DEPTH,
        min_samples_split=DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    return dt
