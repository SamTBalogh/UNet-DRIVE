from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np

from config import DATA_DIR, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample
from ensemble import configure_inference_environment
from evaluate_ensemble import dice_score_numpy
from patch_inference import (
    load_patch_metadata,
    predict_patch_model_probability,
    probability_to_binary_mask,
    resolve_patch_size,
    resolve_stride,
)


def main() -> None:
    args = parse_args()
    configure_inference_environment(device=args.device, cuda_malloc_async=args.cuda_malloc_async)
    tune_thresholds(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune thresholds for patch-trained CV folds with full-image sliding-window validation."
    )
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_patch128_balanced"),
        help="Directory containing fold_X.keras and fold_X_metadata.json files.",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "results" / "cv_bce_dice_patch128_balanced"),
        help="Directory where threshold search CSV files are saved.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated probability thresholds to evaluate.",
    )
    parser.add_argument("--patch-size", type=int, default=128, help="Fallback patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride. Defaults to patch_size / 2.")
    parser.add_argument("--predict-batch-size", type=int, default=16, help="Patch prediction batch size.")
    parser.add_argument("--apply-fov", action="store_true", help="Force probabilities outside FoV to background.")
    parser.add_argument("--tta", action="store_true", help="Use reversible full-image flip TTA for validation.")
    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=None,
        help="Optional per-fold limit for smoke tests. Default evaluates every validation image.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during validation analysis.",
    )
    parser.add_argument("--cuda-malloc-async", action="store_true", help="Set TF_GPU_ALLOCATOR=cuda_malloc_async.")
    return parser.parse_args()


def tune_thresholds(args: argparse.Namespace) -> None:
    import keras

    thresholds = parse_thresholds(args.thresholds)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_fold = load_patch_metadata(models_dir)
    if not metadata_by_fold:
        raise FileNotFoundError(f"No fold metadata files found in {models_dir}")

    samples = list_drive_samples(args.data_dir, split="training", require_manual_2=False)
    by_image_rows: list[dict] = []

    for fold_name, metadata in sorted(metadata_by_fold.items()):
        model_path = models_dir / f"{fold_name}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {fold_name}: {model_path}")

        model = keras.models.load_model(model_path, compile=False)
        patch_size = resolve_patch_size([model], metadata={fold_name: metadata}, fallback=args.patch_size)
        stride = resolve_stride(args.stride, patch_size=patch_size)
        validation_indices = metadata["validation_indices"]
        if args.max_validation_images is not None:
            validation_indices = validation_indices[: args.max_validation_images]
        print(
            f"Evaluating {fold_name}: {len(validation_indices)} validation images, "
            f"patch_size={patch_size}, stride={stride}, TTA={'yes' if args.tta else 'no'}"
        )

        for sample_index in validation_indices:
            sample = samples[sample_index]
            arrays = load_drive_sample(sample)
            manual_1 = binarize_mask(arrays["manual_1"])
            probability = predict_patch_model_probability(
                model=model,
                image=arrays["image"],
                patch_size=patch_size,
                stride=stride,
                batch_size=args.predict_batch_size,
                tta=args.tta,
            )
            if args.apply_fov:
                fov = binarize_mask(arrays["fov_mask"])
                probability = probability * fov.astype(np.float32)

            positive_ratio_manual_1 = float(np.mean(manual_1 > 0))
            for threshold in thresholds:
                prediction = probability_to_binary_mask(probability, threshold=threshold)
                by_image_rows.append(
                    {
                        "fold": fold_name,
                        "model": model_path.name,
                        "image_id": sample.sample_id,
                        "sample_index": sample_index,
                        "threshold": f"{threshold:.2f}",
                        "patch_size": patch_size,
                        "stride": stride,
                        "tta": args.tta,
                        "dice_manual_1": f"{dice_score_numpy(manual_1, prediction):.12f}",
                        "positive_ratio_prediction": f"{float(np.mean(prediction > 0)):.12f}",
                        "positive_ratio_manual_1": f"{positive_ratio_manual_1:.12f}",
                        "probability_min": f"{float(np.min(probability)):.12f}",
                        "probability_mean": f"{float(np.mean(probability)):.12f}",
                        "probability_max": f"{float(np.max(probability)):.12f}",
                    }
                )
        clear_keras_session()

    by_fold_rows = summarize_by_fold(by_image_rows)
    global_rows = summarize_global(by_fold_rows)
    write_csv(output_dir / "threshold_search_by_image.csv", by_image_rows)
    write_csv(output_dir / "threshold_search_by_fold.csv", by_fold_rows)
    write_csv(output_dir / "threshold_search_summary.csv", global_rows)

    best = next((row for row in global_rows if row["is_best_threshold"] == "yes"), None)
    if best:
        print(
            "Best validation threshold: "
            f"{best['threshold']} "
            f"(mean DICE={best['mean_cv_dice_manual_1']})"
        )
    print(f"Saved threshold search outputs to {output_dir}")


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def summarize_by_fold(by_image_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in by_image_rows:
        grouped.setdefault((row["fold"], row["threshold"]), []).append(row)

    summary = []
    for (fold, threshold), rows in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        dice_values = np.asarray([float(row["dice_manual_1"]) for row in rows], dtype=np.float64)
        pred_ratios = np.asarray([float(row["positive_ratio_prediction"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "fold": fold,
                "model": rows[0]["model"],
                "threshold": threshold,
                "patch_size": rows[0]["patch_size"],
                "stride": rows[0]["stride"],
                "tta": rows[0]["tta"],
                "n_images": len(rows),
                "mean_dice_manual_1": f"{float(np.mean(dice_values)):.12f}",
                "std_dice_manual_1": f"{float(np.std(dice_values)):.12f}",
                "min_dice_manual_1": f"{float(np.min(dice_values)):.12f}",
                "max_dice_manual_1": f"{float(np.max(dice_values)):.12f}",
                "mean_positive_ratio_prediction": f"{float(np.mean(pred_ratios)):.12f}",
            }
        )
    return summary


def summarize_global(by_fold_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in by_fold_rows:
        grouped.setdefault(row["threshold"], []).append(row)

    summary = []
    for threshold, rows in sorted(grouped.items(), key=lambda item: float(item[0])):
        fold_means = np.asarray([float(row["mean_dice_manual_1"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "threshold": threshold,
                "n_folds": len(rows),
                "patch_size": rows[0]["patch_size"],
                "stride": rows[0]["stride"],
                "tta": rows[0]["tta"],
                "mean_cv_dice_manual_1": f"{float(np.mean(fold_means)):.12f}",
                "std_cv_dice_manual_1": f"{float(np.std(fold_means)):.12f}",
                "min_fold_dice_manual_1": f"{float(np.min(fold_means)):.12f}",
                "max_fold_dice_manual_1": f"{float(np.max(fold_means)):.12f}",
            }
        )

    if summary:
        best = max(summary, key=lambda row: float(row["mean_cv_dice_manual_1"]))
        for row in summary:
            row["is_best_threshold"] = "yes" if row is best else "no"
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clear_keras_session() -> None:
    try:
        import keras

        keras.backend.clear_session()
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
