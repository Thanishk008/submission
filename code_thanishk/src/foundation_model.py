"""
Foundation Model — Pre-trained MobileNetV2 for Credit Card Fraud Detection
===========================================================================
Reshapes 30 tabular features into a 6x5 single-channel image, tiles to 3
channels, resizes to 32x32x3, and fine-tunes a pre-trained MobileNetV2
(ImageNet weights) for binary fraud classification.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def reshape_to_image(X, img_h=6, img_w=5):
    """Reshape flat feature vectors into single-channel images.

    30 features → (6, 5, 1).  Pad with zeros if feature count != h*w.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1] if X.ndim > 1 else X.shape[0]
    target = img_h * img_w

    if n_features < target:
        pad = np.zeros((n_samples, target - n_features))
        X = np.hstack([X, pad])
    elif n_features > target:
        X = X[:, :target]

    X_img = X.reshape(n_samples, img_h, img_w, 1)
    # Tile to 3 channels (MobileNetV2 expects RGB)
    X_img = np.tile(X_img, (1, 1, 1, 3))
    return X_img


def build_foundation_model(input_shape=(6, 5, 3)):
    """Build a MobileNetV2-based model with frozen backbone + trainable head.

    Args:
        input_shape: Raw input image shape (H, W, 3) before resizing.

    Returns:
        Keras model (uncompiled).
    """
    mobilenet_input_shape = (32, 32, 3)

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=mobilenet_input_shape,
        alpha=0.35,
        pooling='avg',
    )
    # Freeze all backbone layers
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # Resize from raw dims to 32×32 so MobileNetV2 conv layers work
        layers.Resizing(32, 32),
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model


def prepare_foundation_data(X_train, X_test, img_h=6, img_w=5):
    """Convert flat arrays to images for the foundation model.

    Returns reshaped train/test arrays ready for model.fit / model.predict.
    """
    X_train_img = reshape_to_image(X_train, img_h, img_w)
    X_test_img = reshape_to_image(X_test, img_h, img_w)
    return X_train_img, X_test_img
