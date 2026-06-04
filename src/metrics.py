from __future__ import annotations

import numpy as np
from keras import losses, ops, saving


def dice_score_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute thresholded DICE score with NumPy arrays."""

    y_true_bin = (np.asarray(y_true) > 0.5).astype(np.float32).ravel()
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(np.float32).ravel()

    intersection = np.sum(y_true_bin * y_pred_bin)
    denominator = np.sum(y_true_bin) + np.sum(y_pred_bin)
    return float((2.0 * intersection + smooth) / (denominator + smooth))


@saving.register_keras_serializable(package="DriveUNet")
def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Soft DICE coefficient for Keras training logs."""

    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))

    intersection = ops.sum(y_true * y_pred)
    denominator = ops.sum(y_true) + ops.sum(y_pred)
    return (2.0 * intersection + smooth) / (denominator + smooth)


@saving.register_keras_serializable(package="DriveUNet")
def dice_loss(y_true, y_pred):
    """Loss derived from soft DICE."""

    return 1.0 - dice_coef(y_true, y_pred)


@saving.register_keras_serializable(package="DriveUNet")
def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy plus Dice loss for imbalanced vessel masks."""

    bce = losses.binary_crossentropy(y_true, y_pred)
    return ops.mean(bce) + dice_loss(y_true, y_pred)
