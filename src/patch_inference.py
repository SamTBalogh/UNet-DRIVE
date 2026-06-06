from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data import DriveSample, binarize_mask, load_drive_sample
from ensemble import find_model_paths, load_models
from patches import prepare_image


PATCH_TTA_TRANSFORMS = ("identity", "flip_h", "flip_v", "flip_hv")


def load_patch_models(models_dir: str | Path, pattern: str = "fold_*.keras") -> tuple[list[Any], list[Path]]:
    model_paths = find_model_paths(models_dir, pattern=pattern)
    models = load_models(model_paths)
    return models, model_paths


def load_patch_metadata(models_dir: str | Path) -> dict[str, dict]:
    metadata = {}
    for metadata_path in sorted(Path(models_dir).glob("fold_*_metadata.json")):
        fold_name = metadata_path.name.replace("_metadata.json", "")
        metadata[fold_name] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata


def resolve_patch_size(
    models: list[Any],
    metadata: dict[str, dict] | None = None,
    fallback: int = 128,
) -> int:
    if metadata:
        first = next(iter(metadata.values()))
        if "patch_size" in first:
            return int(first["patch_size"])
        if "image_size" in first:
            return int(first["image_size"][0])

    if models:
        input_shape = getattr(models[0], "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape and len(input_shape) >= 3 and input_shape[1]:
            return int(input_shape[1])
    return fallback


def resolve_stride(stride: int | None, patch_size: int) -> int:
    if stride is None:
        return patch_size // 2
    if stride <= 0:
        raise ValueError("stride must be positive")
    return stride


def predict_patch_ensemble_probability(
    models: list[Any],
    sample: DriveSample,
    patch_size: int,
    stride: int,
    batch_size: int = 16,
    apply_fov: bool = False,
    tta: bool = False,
) -> np.ndarray:
    arrays = load_drive_sample(sample)
    image = prepare_image(arrays["image"])
    probabilities = []
    for model in models:
        probability = predict_patch_model_probability(
            model=model,
            image=image,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            tta=tta,
        )
        probabilities.append(probability)

    ensemble_probability = np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)
    if apply_fov:
        fov = binarize_mask(arrays["fov_mask"])
        ensemble_probability = ensemble_probability * fov.astype(np.float32)
    return ensemble_probability


def predict_patch_model_probability(
    model: Any,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    batch_size: int = 16,
    tta: bool = False,
) -> np.ndarray:
    if not tta:
        return predict_sliding_window_probability(
            model=model,
            image=image,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
        )

    probabilities = []
    for transform in PATCH_TTA_TRANSFORMS:
        transformed = apply_image_transform(image, transform)
        probability = predict_sliding_window_probability(
            model=model,
            image=transformed,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
        )
        probabilities.append(invert_probability_transform(probability, transform))
    return np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)


def predict_sliding_window_probability(
    model: Any,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    batch_size: int = 16,
) -> np.ndarray:
    image = prepare_image(image)
    original_height, original_width = image.shape[:2]
    padded = pad_image_to_minimum(image, patch_size=patch_size)
    height, width = padded.shape[:2]
    row_positions = axis_positions(height, patch_size, stride)
    col_positions = axis_positions(width, patch_size, stride)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    patches = []
    locations = []
    for top in row_positions:
        for left in col_positions:
            patch = padded[top : top + patch_size, left : left + patch_size, :]
            patches.append(patch)
            locations.append((top, left))
            if len(patches) == batch_size:
                accumulate_predictions(model, patches, locations, probability_sum, count_map)
                patches = []
                locations = []

    if patches:
        accumulate_predictions(model, patches, locations, probability_sum, count_map)

    count_map[count_map == 0] = 1.0
    probability = probability_sum / count_map
    return probability[:original_height, :original_width].astype(np.float32)


def accumulate_predictions(
    model: Any,
    patches: list[np.ndarray],
    locations: list[tuple[int, int]],
    probability_sum: np.ndarray,
    count_map: np.ndarray,
) -> None:
    batch = np.stack(patches).astype(np.float32)
    predictions = model.predict(batch, verbose=0)[..., 0].astype(np.float32)
    for prediction, (top, left) in zip(predictions, locations):
        height, width = prediction.shape[:2]
        probability_sum[top : top + height, left : left + width] += prediction
        count_map[top : top + height, left : left + width] += 1.0


def pad_image_to_minimum(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return image
    return np.pad(
        image,
        [(0, pad_bottom), (0, pad_right), (0, 0)],
        mode="constant",
        constant_values=0,
    ).astype(np.float32)


def axis_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def apply_image_transform(image: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return image
    if transform == "flip_h":
        return np.ascontiguousarray(np.flip(image, axis=1))
    if transform == "flip_v":
        return np.ascontiguousarray(np.flip(image, axis=0))
    if transform == "flip_hv":
        return np.ascontiguousarray(np.flip(image, axis=(0, 1)))
    raise ValueError(f"Unknown TTA transform: {transform}")


def invert_probability_transform(probability: np.ndarray, transform: str) -> np.ndarray:
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


def save_binary_png(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(output_path)
