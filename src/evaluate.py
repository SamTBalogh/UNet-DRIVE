from __future__ import annotations

import argparse
import csv
from pathlib import Path

import keras
import numpy as np

from config import DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample
from metrics import dice_score_numpy
from predict import infer_resize_strategy_for_model, predict_sample_mask, resolve_model_image_size


def main() -> None:
    args = parse_args()

    model = keras.models.load_model(args.model)
    resize_strategy = infer_resize_strategy_for_model(args.model, requested=args.resize_strategy)
    image_size = resolve_model_image_size(model, fallback=(args.image_height, args.image_width))
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test")
    rows = evaluate_samples(
        model=model,
        samples=samples,
        image_size=image_size,
        resize_strategy=resize_strategy,
        threshold=args.threshold,
        apply_fov=args.apply_fov,
        model_name=Path(args.model).name,
    )

    output_path = Path(args.output)
    save_rows(rows, output_path)

    dice_values = [row["dice_mean"] for row in rows]
    print(f"Saved evaluation to {output_path}")
    print(f"Mean DICE: {np.mean(dice_values):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model with DICE score.")
    parser.add_argument("--model", required=True, help="Path to a .keras model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to evaluate.")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "results" / "evaluation.csv"), help="CSV output path.")
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


def evaluate_samples(
    model: keras.Model,
    samples: list,
    image_size: tuple[int, int],
    resize_strategy: str,
    threshold: float,
    apply_fov: bool,
    model_name: str,
) -> list[dict[str, str | float]]:
    rows = []

    for sample in samples:
        prediction = predict_sample_mask(
            model=model,
            sample=sample,
            image_size=image_size,
            resize_strategy=resize_strategy,
            threshold=threshold,
            apply_fov=apply_fov,
        )
        arrays = load_drive_sample(sample)
        manual_1 = binarize_mask(arrays["manual_1"])
        dice_manual_1 = dice_score_numpy(manual_1, prediction, threshold=0.5)

        dice_manual_2 = ""
        dice_values = [dice_manual_1]
        if arrays["manual_2"] is not None:
            manual_2 = binarize_mask(arrays["manual_2"])
            dice_manual_2 = dice_score_numpy(manual_2, prediction, threshold=0.5)
            dice_values.append(dice_manual_2)

        rows.append(
            {
                "image_id": sample.sample_id,
                "model": model_name,
                "threshold": threshold,
                "dice_manual_1": dice_manual_1,
                "dice_manual_2": dice_manual_2,
                "dice_mean": float(np.mean(dice_values)),
            }
        )

    return rows


def save_rows(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "model", "threshold", "dice_manual_1", "dice_manual_2", "dice_mean"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
