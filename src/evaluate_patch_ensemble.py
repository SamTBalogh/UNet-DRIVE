from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from config import DATA_DIR, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample
from ensemble import configure_inference_environment
from evaluate_ensemble import default_summary_path, dice_score_numpy
from patch_inference import (
    load_patch_metadata,
    load_patch_models,
    predict_patch_ensemble_probability,
    probability_to_binary_mask,
    resolve_patch_size,
    resolve_stride,
)
from postprocess import remove_small_components


def main() -> None:
    args = parse_args()
    configure_inference_environment(device=args.device, cuda_malloc_async=args.cuda_malloc_async)

    models, model_paths = load_patch_models(args.models_dir, pattern=args.model_pattern)
    metadata = load_patch_metadata(args.models_dir)
    patch_size = resolve_patch_size(models, metadata=metadata, fallback=args.patch_size)
    stride = resolve_stride(args.stride, patch_size=patch_size)
    model_names = [path.name for path in model_paths]

    samples = select_samples(
        list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test"),
        sample_ids=parse_sample_ids(args.sample_ids),
        max_samples=args.max_samples,
    )
    rows = evaluate_samples(
        models=models,
        model_names=model_names,
        samples=samples,
        patch_size=patch_size,
        stride=stride,
        batch_size=args.predict_batch_size,
        threshold=args.threshold,
        postprocess_min_size=args.postprocess_min_size,
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
    print(f"Loaded {len(models)} patch models: {', '.join(model_names)}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Prediction batch size: {args.predict_batch_size}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"Postprocess min component size: {args.postprocess_min_size}")
    print(f"Saved evaluation to {output_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Mean DICE: {np.mean(dice_values):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a patch-trained ensemble with sliding-window inference.")
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_patch128_balanced"),
        help="Directory containing patch-trained fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to evaluate.")
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "results" / "patch_ensemble_test.csv"),
        help="Per-image CSV output path.",
    )
    parser.add_argument("--summary-output", default=None, help="Optional summary CSV output path.")
    parser.add_argument("--ensemble-name", default="patch_ensemble", help="Name written in the output CSV.")
    parser.add_argument("--patch-size", type=int, default=128, help="Fallback patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride. Defaults to patch_size / 2.")
    parser.add_argument("--predict-batch-size", type=int, default=16, help="Patch prediction batch size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel pixels.")
    parser.add_argument(
        "--postprocess-min-size",
        type=int,
        default=0,
        help="Minimum connected-component size to keep after thresholding. Use 0 for no postprocessing.",
    )
    parser.add_argument("--apply-fov", action="store_true", help="Force pixels outside DRIVE FoV to background.")
    parser.add_argument("--tta", action="store_true", help="Use reversible full-image flip TTA before thresholding.")
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample ids. Empty means all samples unless --max-samples is set.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit selected samples.")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during evaluation.",
    )
    parser.add_argument("--cuda-malloc-async", action="store_true", help="Set TF_GPU_ALLOCATOR=cuda_malloc_async.")
    return parser.parse_args()


def parse_sample_ids(raw: str) -> list[str] | None:
    sample_ids = [value.strip() for value in raw.split(",") if value.strip()]
    return sample_ids or None


def select_samples(samples: list, sample_ids: list[str] | None, max_samples: int | None) -> list:
    if sample_ids:
        by_id = {sample.sample_id: sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"Sample ids not found: {', '.join(missing)}")
        selected = [by_id[sample_id] for sample_id in sample_ids]
    else:
        selected = samples

    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def evaluate_samples(
    models: list,
    model_names: list[str],
    samples: list,
    patch_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    postprocess_min_size: int,
    apply_fov: bool,
    tta: bool,
    ensemble_name: str,
) -> list[dict[str, str | float]]:
    rows = []
    model_list = ";".join(model_names)

    for sample in samples:
        probability = predict_patch_ensemble_probability(
            models=models,
            sample=sample,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            apply_fov=apply_fov,
            tta=tta,
        )
        prediction = probability_to_binary_mask(probability, threshold=threshold)

        arrays = load_drive_sample(sample)
        fov = binarize_mask(arrays["fov_mask"]) if apply_fov else None
        prediction = remove_small_components(
            prediction,
            min_size=postprocess_min_size,
            fov_mask=fov,
        )
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
                "postprocess_min_size": postprocess_min_size,
                "patch_size": patch_size,
                "stride": stride,
                "tta": tta,
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
        "postprocess_min_size",
        "patch_size",
        "stride",
        "tta",
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
        "postprocess_min_size": rows[0]["postprocess_min_size"] if rows else "",
        "patch_size": rows[0]["patch_size"] if rows else "",
        "stride": rows[0]["stride"] if rows else "",
        "tta": rows[0]["tta"] if rows else "",
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
        "postprocess_min_size",
        "patch_size",
        "stride",
        "tta",
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


if __name__ == "__main__":
    main()
