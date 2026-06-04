from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune probability thresholds on the saved cross-validation folds, "
            "using each fold's validation indices from metadata JSON files."
        )
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/models/cv_5folds",
        help="Directory containing fold_X.keras and fold_X_metadata.json files.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/DRIVE",
        help="Path to DRIVE root folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/results/cv_5folds",
        help="Directory where threshold search CSV files are saved.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated probability thresholds to evaluate.",
    )
    parser.add_argument("--image-height", type=int, default=512, help="Fallback model input height.")
    parser.add_argument("--image-width", type=int, default=512, help="Fallback model input width.")
    parser.add_argument(
        "--resize-strategy",
        choices=("resize", "pad"),
        default=None,
        help="Preprocessing strategy. Defaults to metadata value when available.",
    )
    parser.add_argument("--pad-multiple", type=int, default=16, help="Pad multiple kept for metadata traceability.")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during validation analysis.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help="Set TF_GPU_ALLOCATOR=cuda_malloc_async before importing Keras/TensorFlow.",
    )
    parser.add_argument(
        "--apply-fov",
        action="store_true",
        help="Force probabilities outside the DRIVE FoV mask to background before thresholding.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.cuda_malloc_async and args.device != "cpu":
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def load_fold_metadata(models_dir: Path) -> list[dict]:
    metadata_files = sorted(models_dir.glob("fold_*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No fold metadata files found in {models_dir}")

    metadata = []
    for metadata_path in metadata_files:
        fold_name = metadata_path.name.replace("_metadata.json", "")
        model_path = models_dir / f"{fold_name}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {fold_name}: {model_path}")
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        data["fold"] = fold_name
        data["model_path"] = model_path
        data["metadata_path"] = metadata_path
        metadata.append(data)
    return metadata


def resize_probability(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    with Image.fromarray(probability.astype(np.float32), mode="F") as image:
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def dice_binary(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true = (np.asarray(y_true) > 0).astype(np.float32).ravel()
    y_pred = (np.asarray(y_pred) > 0).astype(np.float32).ravel()
    intersection = float(np.sum(y_true * y_pred))
    denominator = float(np.sum(y_true) + np.sum(y_pred))
    return (2.0 * intersection + smooth) / (denominator + smooth)


def predict_probability_original_size(
    model,
    sample,
    image_size: tuple[int, int],
    resize_strategy: str,
    apply_fov: bool,
) -> np.ndarray:
    from data import binarize_mask, crop_array_from_padded, load_drive_sample, load_preprocessed_sample

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

    if apply_fov:
        fov = binarize_mask(original["fov_mask"])
        probability = probability * fov.astype(np.float32)

    return probability


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_fold(by_image_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in by_image_rows:
        key = (row["fold"], row["threshold"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (fold, threshold), rows in sorted(grouped.items()):
        dice_values = np.asarray([float(row["dice_manual_1"]) for row in rows], dtype=np.float64)
        pred_ratios = np.asarray(
            [float(row["positive_ratio_prediction"]) for row in rows], dtype=np.float64
        )
        summary.append(
            {
                "fold": fold,
                "model": rows[0]["model"],
                "threshold": f"{threshold:.2f}",
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
        fold_means = np.asarray(
            [float(row["mean_dice_manual_1"]) for row in rows], dtype=np.float64
        )
        summary.append(
            {
                "threshold": threshold,
                "n_folds": len(rows),
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


def clear_keras_session() -> None:
    try:
        import keras

        keras.backend.clear_session()
    finally:
        gc.collect()


def tune_thresholds(args: argparse.Namespace) -> None:
    configure_environment(args)

    import keras
    from data import binarize_mask, list_drive_samples, load_drive_sample
    import metrics  # noqa: F401 - register custom dice_coef before load_model.

    thresholds = parse_thresholds(args.thresholds)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    fold_metadata = load_fold_metadata(models_dir)
    resize_strategy, image_size = resolve_preprocessing(args, fold_metadata)
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    samples = list_drive_samples(args.data_dir, split="training", require_manual_2=False)

    by_image_rows: list[dict] = []

    for metadata in fold_metadata:
        fold = metadata["fold"]
        model_path = metadata["model_path"]
        validation_indices = metadata["validation_indices"]

        print(f"Evaluating {fold}: {len(validation_indices)} validation images")
        model = keras.models.load_model(model_path)

        for sample_index in validation_indices:
            sample = samples[sample_index]
            original = load_drive_sample(sample)
            manual_1 = binarize_mask(original["manual_1"])
            probability = predict_probability_original_size(
                model=model,
                sample=sample,
                image_size=image_size,
                resize_strategy=resize_strategy,
                apply_fov=args.apply_fov,
            )
            positive_ratio_manual_1 = float(np.mean(manual_1 > 0))

            for threshold in thresholds:
                prediction = (probability >= threshold).astype(np.uint8)
                by_image_rows.append(
                    {
                        "fold": fold,
                        "model": model_path.name,
                        "image_id": sample.sample_id,
                        "sample_index": sample_index,
                        "threshold": threshold,
                        "dice_manual_1": f"{dice_binary(manual_1, prediction):.12f}",
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


def resolve_preprocessing(args: argparse.Namespace, fold_metadata: list[dict]) -> tuple[str, tuple[int, int]]:
    first_metadata = fold_metadata[0]
    resize_strategy = args.resize_strategy or first_metadata.get("resize_strategy", "resize")
    if "image_size" in first_metadata:
        image_size = tuple(int(value) for value in first_metadata["image_size"])
    else:
        image_size = (args.image_height, args.image_width)
    return resize_strategy, image_size


def main() -> None:
    args = parse_args()
    tune_thresholds(args)


if __name__ == "__main__":
    main()
