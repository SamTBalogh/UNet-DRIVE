from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import IMAGE_SIZE
from data import (
    DriveSample,
    binarize_mask,
    crop_array_from_padded,
    load_drive_sample,
    load_preprocessed_sample,
)


TTA_TRANSFORMS = ("identity", "flip_h", "flip_v", "flip_hv")


def configure_inference_environment(device: str = "cpu", cuda_malloc_async: bool = False) -> None:
    """Configure TensorFlow-related environment variables before importing Keras."""

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cuda_malloc_async:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def find_model_paths(models_dir: str | Path, pattern: str = "fold_*.keras") -> list[Path]:
    """Return sorted Keras model paths for an ensemble."""

    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory not found: {root}")

    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No models matching '{pattern}' found in {root}")
    return paths


def infer_resize_strategy(models_dir: str | Path, requested: str | None = None) -> str:
    if requested:
        return requested
    metadata_files = sorted(Path(models_dir).glob("fold_*_metadata.json"))
    if metadata_files:
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        return metadata.get("resize_strategy", "resize")
    return "resize"


def load_models(model_paths: list[Path]) -> list[Any]:
    """Load models for inference without compiling training losses or metrics."""

    import keras

    return [keras.models.load_model(path, compile=False) for path in model_paths]


def resolve_models_image_size(models: list[Any], fallback: tuple[int, int] = IMAGE_SIZE) -> tuple[int, int]:
    if not models:
        return fallback
    input_shape = getattr(models[0], "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])
    return fallback


def predict_ensemble_probability(
    models: list[Any],
    sample: DriveSample,
    image_size: tuple[int, int] = IMAGE_SIZE,
    resize_strategy: str = "resize",
    apply_fov: bool = False,
    tta: bool = False,
) -> np.ndarray:
    """Average model probabilities and return them in the original image size."""

    preprocessed = load_preprocessed_sample(
        sample,
        image_size=image_size,
        resize_strategy=resize_strategy,
    )
    original = load_drive_sample(sample)
    original_height, original_width = original["image"].shape[:2]
    batch = preprocessed["image"][np.newaxis, ...]

    probabilities = []
    for model in models:
        probability = predict_model_probability(model, batch=batch, tta=tta)
        if resize_strategy == "pad":
            probability = crop_array_from_padded(probability, original_size=(original_height, original_width))
        else:
            probability = resize_probability(probability, size=(original_height, original_width))
        probabilities.append(probability)

    ensemble_probability = np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)
    if apply_fov:
        fov = binarize_mask(original["fov_mask"])
        ensemble_probability = ensemble_probability * fov.astype(np.float32)
    return ensemble_probability


def predict_model_probability(model: Any, batch: np.ndarray, tta: bool = False) -> np.ndarray:
    """Predict one model probability map, optionally averaging reversible flip TTA."""

    if not tta:
        return model.predict(batch, verbose=0)[0, ..., 0].astype(np.float32)

    probabilities = []
    for transform in TTA_TRANSFORMS:
        transformed_batch = apply_tta_transform(batch, transform)
        probability = model.predict(transformed_batch, verbose=0)[0, ..., 0]
        probabilities.append(invert_tta_probability(probability, transform))
    return np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)


def apply_tta_transform(batch: np.ndarray, transform: str) -> np.ndarray:
    """Apply a reversible TTA transform to a NHWC batch."""

    if transform == "identity":
        return batch
    if transform == "flip_h":
        return np.ascontiguousarray(np.flip(batch, axis=2))
    if transform == "flip_v":
        return np.ascontiguousarray(np.flip(batch, axis=1))
    if transform == "flip_hv":
        return np.ascontiguousarray(np.flip(batch, axis=(1, 2)))
    raise ValueError(f"Unknown TTA transform: {transform}")


def invert_tta_probability(probability: np.ndarray, transform: str) -> np.ndarray:
    """Map a TTA probability prediction back to the original orientation."""

    if transform == "identity":
        return probability.astype(np.float32)
    if transform == "flip_h":
        return np.flip(probability, axis=1).astype(np.float32)
    if transform == "flip_v":
        return np.flip(probability, axis=0).astype(np.float32)
    if transform == "flip_hv":
        return np.flip(probability, axis=(0, 1)).astype(np.float32)
    raise ValueError(f"Unknown TTA transform: {transform}")


def probability_to_binary_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (probability >= threshold).astype(np.uint8)


def resize_probability(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    with Image.fromarray(probability.astype(np.float32)) as image:
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def save_binary_png(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image.save(output_path)
