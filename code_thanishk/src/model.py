"""
1D CNN Model Architecture for Credit Card Fraud Detection

Supports configurable dropout rate and batch normalization for ablation studies.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape=(30, 1), dropout_rate=0.3, use_batchnorm=True):
    """Build and return the 1D CNN model.

    Args:
        input_shape: Shape of each input sample (features, 1).
        dropout_rate: Dropout probability after each block (0.0 = no dropout).
        use_batchnorm: Whether to include BatchNormalization layers.
    """
    model_layers = [layers.Input(shape=input_shape)]

    # --- Conv Block 1 ---
    model_layers.append(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    if use_batchnorm:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model_layers.append(layers.Dropout(dropout_rate))

    # --- Conv Block 2 ---
    model_layers.append(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    if use_batchnorm:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model_layers.append(layers.Dropout(dropout_rate))

    # --- Conv Block 3 ---
    model_layers.append(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    if use_batchnorm:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.MaxPooling1D(pool_size=2))
    if dropout_rate > 0:
        model_layers.append(layers.Dropout(dropout_rate))

    # --- Dense Head ---
    model_layers.append(layers.Flatten())

    model_layers.append(layers.Dense(256, activation='relu'))
    if use_batchnorm:
        model_layers.append(layers.BatchNormalization())
    dense_dropout = min(dropout_rate + 0.1, 0.5) if dropout_rate > 0 else 0
    if dense_dropout > 0:
        model_layers.append(layers.Dropout(dense_dropout))

    model_layers.append(layers.Dense(128, activation='relu'))
    if use_batchnorm:
        model_layers.append(layers.BatchNormalization())
    if dense_dropout > 0:
        model_layers.append(layers.Dropout(dense_dropout))

    model_layers.append(layers.Dense(1, activation='sigmoid'))

    model = keras.Sequential(model_layers)
    return model
