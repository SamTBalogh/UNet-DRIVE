from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from config import DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR
from data import (
    binarize_mask,
    crop_array_from_padded,
    list_drive_samples,
    load_drive_sample,
    load_preprocessed_sample,
)
from metrics import dice_coef


def main() -> None:
    args = parse_args()

    model = keras.models.load_model(args.model)
    resize_strategy = infer_resize_strategy_for_model(args.model, requested=args.resize_strategy)
    image_size = resolve_model_image_size(model, fallback=(args.image_height, args.image_width))
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        mask = predict_sample_mask(
            model=model,
            sample=sample,
            image_size=image_size,
            resize_strategy=resize_strategy,
            threshold=args.threshold,
            apply_fov=args.apply_fov,
        )
        output_path = output_dir / f"{sample.sample_id}_{args.split}_segmentation.png"
        save_binary_png(mask, output_path)
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG vessel segmentations with a trained model.")
    parser.add_argument("--model", required=True, help="Path to a .keras model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to predict.")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "segmentations"), help="Where PNG masks are saved.")
    parser.add_argument("--image-height", type=int, default=IMAGE_SIZE[0], help="Model input height.")
    parser.add_argument("--image-width", type=int, default=IMAGE_SIZE[1], help="Model input width.")
    parser.add_argument(
        "--resize-strategy",
        choices=("resize", "pad"),
        default=None,
        help="Preprocessing strategy. Defaults to sibling metadata value when available.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel pixels.")
    parser.add_argument("--apply-fov", action="store_true", help="Force pixels outside the DRIVE FoV mask to background.")
    return parser.parse_args()


def predict_sample_mask(
    model: keras.Model,
    sample,
    image_size: tuple[int, int] = IMAGE_SIZE,
    resize_strategy: str = "resize",
    threshold: float = 0.5,
    apply_fov: bool = False,
) -> np.ndarray:
    """Predict a binary mask in the original DRIVE image size."""

    preprocessed = load_preprocessed_sample(
        sample,
        image_size=image_size,
        resize_strategy=resize_strategy,
    )
    original = load_drive_sample(sample)
    original_height, original_width = original["image"].shape[:2]

    probability = model.predict(preprocessed["image"][np.newaxis, ...], verbose=0)[0, ..., 0]
    if resize_strategy == "pad":
        probability = crop_array_from_padded(probability, original_size=(original_height, original_width))
    else:
        probability = resize_probability(probability, size=(original_height, original_width))
    binary = (probability >= threshold).astype(np.uint8)

    if apply_fov:
        fov = binarize_mask(original["fov_mask"])
        binary = (binary * fov).astype(np.uint8)

    return binary


def resolve_model_image_size(model: keras.Model, fallback: tuple[int, int]) -> tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])
    return fallback


def infer_resize_strategy_for_model(model_path: str | Path, requested: str | None = None) -> str:
    if requested:
        return requested
    path = Path(model_path)
    metadata_path = path.with_name(f"{path.stem}_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("resize_strategy", "resize")
    return "resize"


def resize_probability(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    with Image.fromarray(probability.astype(np.float32)) as image:
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    with Image.fromarray((mask > 0).astype(np.uint8) * 255) as image:
        resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
        return (np.asarray(resized) > 127).astype(np.uint8)


def save_binary_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image.save(output_path)


if __name__ == "__main__":
    main()
