from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from config import DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample
from ensemble import (
    configure_inference_environment,
    find_model_paths,
    infer_resize_strategy,
    load_models,
    predict_ensemble_probability,
    probability_to_binary_mask,
    resolve_models_image_size,
)


def main() -> None:
    args = parse_args()
    configure_inference_environment(device=args.device, cuda_malloc_async=args.cuda_malloc_async)

    resize_strategy = infer_resize_strategy(args.models_dir, requested=args.resize_strategy)
    model_paths = find_model_paths(args.models_dir, pattern=args.model_pattern)
    models = load_models(model_paths)
    image_size = resolve_models_image_size(models, fallback=(args.image_height, args.image_width))
    model_names = [path.name for path in model_paths]

    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test")
    rows = evaluate_samples(
        models=models,
        model_names=model_names,
        samples=samples,
        image_size=image_size,
        resize_strategy=resize_strategy,
        threshold=args.threshold,
        apply_fov=args.apply_fov,
        tta=args.tta,
        ensemble_name=args.ensemble_name,
    )

    output_path = Path(args.output)
    save_rows(rows, output_path)

    summary_path = Path(args.summary_output) if args.summary_output else default_summary_path(output_path)
    summary_rows = summarize_rows(rows, model_names=model_names, ensemble_name=args.ensemble_name)
    save_summary(summary_rows, summary_path)

    dice_values = [float(row["dice_mean"]) for row in rows]
    print(f"Loaded {len(models)} models: {', '.join(model_names)}")
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"Saved evaluation to {output_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Mean DICE: {np.mean(dice_values):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of fold models with DICE score.")
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_flips_valloss"),
        help="Directory containing fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to evaluate.")
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "results" / "ensemble_test.csv"),
        help="Per-image CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary CSV output path. Defaults to '<output_stem>_summary.csv'.",
    )
    parser.add_argument("--ensemble-name", default="ensemble", help="Name written in the output CSV.")
    parser.add_argument("--image-height", type=int, default=IMAGE_SIZE[0], help="Model input height.")
    parser.add_argument("--image-width", type=int, default=IMAGE_SIZE[1], help="Model input width.")
    parser.add_argument(
        "--resize-strategy",
        choices=("resize", "pad"),
        default=None,
        help="Preprocessing strategy. Defaults to metadata value when available.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel pixels.")
    parser.add_argument("--apply-fov", action="store_true", help="Force pixels outside the DRIVE FoV to background.")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average original, horizontal flip, vertical flip and both-flip predictions before thresholding.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during evaluation.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help="Set TF_GPU_ALLOCATOR=cuda_malloc_async when using GPU.",
    )
    return parser.parse_args()


def evaluate_samples(
    models: list,
    model_names: list[str],
    samples: list,
    image_size: tuple[int, int],
    resize_strategy: str,
    threshold: float,
    apply_fov: bool,
    tta: bool,
    ensemble_name: str,
) -> list[dict[str, str | float]]:
    rows = []
    model_list = ";".join(model_names)

    for sample in samples:
        probability = predict_ensemble_probability(
            models=models,
            sample=sample,
            image_size=image_size,
            resize_strategy=resize_strategy,
            apply_fov=apply_fov,
            tta=tta,
        )
        prediction = probability_to_binary_mask(probability, threshold=threshold)

        arrays = load_drive_sample(sample)
        manual_1 = binarize_mask(arrays["manual_1"])
        dice_manual_1 = dice_score_numpy(manual_1, prediction, threshold=0.5)
        positive_ratio_manual_1 = float(np.mean(manual_1 > 0))

        dice_manual_2: str | float = ""
        positive_ratio_manual_2: str | float = ""
        dice_values = [dice_manual_1]
        if arrays["manual_2"] is not None:
            manual_2 = binarize_mask(arrays["manual_2"])
            dice_manual_2 = dice_score_numpy(manual_2, prediction, threshold=0.5)
            positive_ratio_manual_2 = float(np.mean(manual_2 > 0))
            dice_values.append(dice_manual_2)

        rows.append(
            {
                "image_id": sample.sample_id,
                "model": ensemble_name,
                "models": model_list,
                "threshold": threshold,
                "dice_manual_1": dice_manual_1,
                "dice_manual_2": dice_manual_2,
                "dice_mean": float(np.mean(dice_values)),
                "positive_ratio_prediction": float(np.mean(prediction > 0)),
                "positive_ratio_manual_1": positive_ratio_manual_1,
                "positive_ratio_manual_2": positive_ratio_manual_2,
            }
        )

    return rows


def save_rows(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "model",
        "models",
        "threshold",
        "dice_manual_1",
        "dice_manual_2",
        "dice_mean",
        "positive_ratio_prediction",
        "positive_ratio_manual_1",
        "positive_ratio_manual_2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dice_score_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    y_true_bin = (np.asarray(y_true) > 0.5).astype(np.float32).ravel()
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(np.float32).ravel()
    intersection = np.sum(y_true_bin * y_pred_bin)
    denominator = np.sum(y_true_bin) + np.sum(y_pred_bin)
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def summarize_rows(
    rows: list[dict[str, str | float]],
    model_names: list[str],
    ensemble_name: str,
) -> list[dict[str, str | float]]:
    dice_1 = np.asarray([float(row["dice_manual_1"]) for row in rows], dtype=np.float64)
    dice_mean = np.asarray([float(row["dice_mean"]) for row in rows], dtype=np.float64)
    pred_ratio = np.asarray([float(row["positive_ratio_prediction"]) for row in rows], dtype=np.float64)
    dice_2_values = [row["dice_manual_2"] for row in rows if row["dice_manual_2"] != ""]

    summary: dict[str, str | float] = {
        "model": ensemble_name,
        "models": ";".join(model_names),
        "threshold": rows[0]["threshold"] if rows else "",
        "n_images": len(rows),
        "mean_dice_manual_1": float(np.mean(dice_1)) if len(dice_1) else "",
        "std_dice_manual_1": float(np.std(dice_1)) if len(dice_1) else "",
        "mean_dice_manual_2": "",
        "std_dice_manual_2": "",
        "mean_dice_total": float(np.mean(dice_mean)) if len(dice_mean) else "",
        "std_dice_total": float(np.std(dice_mean)) if len(dice_mean) else "",
        "min_dice_total": float(np.min(dice_mean)) if len(dice_mean) else "",
        "max_dice_total": float(np.max(dice_mean)) if len(dice_mean) else "",
        "mean_positive_ratio_prediction": float(np.mean(pred_ratio)) if len(pred_ratio) else "",
    }

    if dice_2_values:
        dice_2 = np.asarray([float(value) for value in dice_2_values], dtype=np.float64)
        summary["mean_dice_manual_2"] = float(np.mean(dice_2))
        summary["std_dice_manual_2"] = float(np.std(dice_2))

    return [summary]


def save_summary(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "models",
        "threshold",
        "n_images",
        "mean_dice_manual_1",
        "std_dice_manual_1",
        "mean_dice_manual_2",
        "std_dice_manual_2",
        "mean_dice_total",
        "std_dice_total",
        "min_dice_total",
        "max_dice_total",
        "mean_positive_ratio_prediction",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")


if __name__ == "__main__":
    main()
