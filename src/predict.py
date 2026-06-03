from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from config import DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample, load_preprocessed_sample
from metrics import dice_coef


def main() -> None:
    args = parse_args()
    image_size = (args.image_height, args.image_width)

    model = keras.models.load_model(args.model)
    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        mask = predict_sample_mask(
            model=model,
            sample=sample,
            image_size=image_size,
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel pixels.")
    parser.add_argument("--apply-fov", action="store_true", help="Force pixels outside the DRIVE FoV mask to background.")
    return parser.parse_args()


def predict_sample_mask(
    model: keras.Model,
    sample,
    image_size: tuple[int, int] = IMAGE_SIZE,
    threshold: float = 0.5,
    apply_fov: bool = False,
) -> np.ndarray:
    """Predict a binary mask in the original DRIVE image size."""

    preprocessed = load_preprocessed_sample(sample, image_size=image_size)
    original = load_drive_sample(sample)
    original_height, original_width = original["image"].shape[:2]

    prediction = model.predict(preprocessed["image"][np.newaxis, ...], verbose=0)[0, ..., 0]
    binary = (prediction >= threshold).astype(np.uint8)
    binary = resize_binary_mask(binary, size=(original_height, original_width))

    if apply_fov:
        fov = binarize_mask(original["fov_mask"])
        binary = (binary * fov).astype(np.uint8)

    return binary


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
