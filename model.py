# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D,
    Flatten, Dense, Dropout
)
from config import SEQ_LEN, EMB_DIM


def model(vocab_size: int) -> tf.keras.Model:
    """
    Xây dựng mô hình CNN 1D gồm 4 lớp chính:
      - Lớp 1: Embedding layer cho đầu vào chuỗi (API+DLL+Mutex).
      - Lớp 2: Block 2 convolution layers (Conv1D) và 2 pooling layers (MaxPooling1D) để trích xuất đặc trưng.
      - Lớp 3: Dense layer (fully connected) để học biểu diễn phi tuyến.
      - Lớp 4: Output layer với activation sigmoid cho nhị phân.
    """
    model = Sequential([
        # Lớp 1: Embedding Layer (Input layer)
        Embedding(input_dim=vocab_size,
                  output_dim=EMB_DIM,
                  input_length=SEQ_LEN),

        # Lớp 2: Convolutional Block - 2 convolutional layers và 2 pooling layers
        # Layer Conv #1
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        Dropout(0.25),
        MaxPooling1D(pool_size=2),
        # Layer Conv #2
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.25),
        MaxPooling1D(pool_size=2),

        # Lớp 3: Fully Connected Layer
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),

        # Lớp 4: Output Layer (Sigmoid Classifier)
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model