from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import IMAGE_SIZE
from data import DriveSample, binarize_mask, load_drive_sample, load_preprocessed_sample


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


def load_models(model_paths: list[Path]) -> list[Any]:
    """Load models for inference without compiling training losses or metrics."""

    import keras

    return [keras.models.load_model(path, compile=False) for path in model_paths]


def predict_ensemble_probability(
    models: list[Any],
    sample: DriveSample,
    image_size: tuple[int, int] = IMAGE_SIZE,
    apply_fov: bool = False,
) -> np.ndarray:
    """Average model probabilities and return them in the original image size."""

    preprocessed = load_preprocessed_sample(sample, image_size=image_size)
    original = load_drive_sample(sample)
    original_height, original_width = original["image"].shape[:2]
    batch = preprocessed["image"][np.newaxis, ...]

    probabilities = []
    for model in models:
        probability = model.predict(batch, verbose=0)[0, ..., 0]
        probability = resize_probability(probability, size=(original_height, original_width))
        probabilities.append(probability)

    ensemble_probability = np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)
    if apply_fov:
        fov = binarize_mask(original["fov_mask"])
        ensemble_probability = ensemble_probability * fov.astype(np.float32)
    return ensemble_probability


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
