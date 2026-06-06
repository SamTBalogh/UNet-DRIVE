from __future__ import annotations

import numpy as np
from keras import losses, ops, saving

WEIGHTED_BCE_POSITIVE_WEIGHT = 4.0
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
FOCAL_TVERSKY_GAMMA = 0.75
THIN_DICE_WEIGHT_FACTOR = 2.0


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


def weighted_binary_crossentropy(
    y_true,
    y_pred,
    positive_weight: float = WEIGHTED_BCE_POSITIVE_WEIGHT,
):
    """Binary cross-entropy with extra weight on vessel pixels."""

    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)

    weights = 1.0 + y_true * (positive_weight - 1.0)
    bce = -(
        y_true * ops.log(y_pred)
        + (1.0 - y_true) * ops.log(1.0 - y_pred)
    )
    return ops.sum(weights * bce) / (ops.sum(weights) + 1e-6)


@saving.register_keras_serializable(package="DriveUNet")
def weighted_bce_dice_loss(y_true, y_pred):
    """Weighted BCE plus Dice loss to reduce missed vessel pixels."""

    return weighted_binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def tversky_coef(
    y_true,
    y_pred,
    alpha: float = TVERSKY_ALPHA,
    beta: float = TVERSKY_BETA,
    smooth: float = 1e-6,
):
    """Soft Tversky coefficient; beta > alpha penalizes false negatives more."""

    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))

    true_positive = ops.sum(y_true * y_pred)
    false_positive = ops.sum((1.0 - y_true) * y_pred)
    false_negative = ops.sum(y_true * (1.0 - y_pred))
    denominator = true_positive + alpha * false_positive + beta * false_negative
    return (true_positive + smooth) / (denominator + smooth)


@saving.register_keras_serializable(package="DriveUNet")
def focal_tversky_loss(y_true, y_pred):
    """Focal Tversky loss biased toward false-negative reduction."""

    return ops.power(1.0 - tversky_coef(y_true, y_pred), FOCAL_TVERSKY_GAMMA)


def thin_vessel_proxy_weights(
    y_true,
    weight_factor: float = THIN_DICE_WEIGHT_FACTOR,
):
    """Approximate thin-vessel pixels from local mask density."""

    y_true = ops.cast(y_true, "float32")
    local_density = ops.average_pool(
        y_true,
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
    )
    thin_proxy = y_true * (1.0 - local_density)
    return 1.0 + weight_factor * thin_proxy


@saving.register_keras_serializable(package="DriveUNet")
def thin_weighted_dice_loss(y_true, y_pred):
    """Dice loss with larger weight on sparse local vessel neighborhoods."""

    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    weights = thin_vessel_proxy_weights(y_true)

    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))
    weights = ops.reshape(weights, (-1,))

    intersection = ops.sum(weights * y_true * y_pred)
    denominator = ops.sum(weights * y_true) + ops.sum(weights * y_pred)
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
