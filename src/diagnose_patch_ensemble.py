from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample
from ensemble import configure_inference_environment
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
    samples = select_samples(
        list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test"),
        sample_ids=parse_sample_ids(args.sample_ids),
        max_samples=args.max_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(models)} patch models: {', '.join(path.name for path in model_paths)}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Prediction batch size: {args.predict_batch_size}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"Postprocess min component size: {args.postprocess_min_size}")
    print(f"Generating diagnostics for {len(samples)} samples")

    for sample in samples:
        output_path = output_dir / f"{sample.sample_id}_{args.split}_patch_diagnostic.png"
        create_diagnostic_figure(
            models=models,
            sample=sample,
            patch_size=patch_size,
            stride=stride,
            batch_size=args.predict_batch_size,
            threshold=args.threshold,
            postprocess_min_size=args.postprocess_min_size,
            apply_fov=args.apply_fov,
            tta=args.tta,
            error_reference=args.error_reference,
            output_path=output_path,
        )
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual diagnostics for a patch-trained ensemble prediction."
    )
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_patch128_balanced"),
        help="Directory containing patch-trained fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to diagnose.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "figures" / "patch_ensemble_diagnostics"),
        help="Where diagnostic figures are saved.",
    )
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
        default="01,03,07,14,20",
        help="Comma-separated sample ids. Use an empty value with --max-samples to select from the start.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit selected samples.")
    parser.add_argument(
        "--error-reference",
        choices=("manual_1", "manual_2"),
        default="manual_1",
        help="Manual mask used to compute false positives and false negatives.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during diagnostics.",
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


def create_diagnostic_figure(
    models: list,
    sample,
    patch_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    postprocess_min_size: int,
    apply_fov: bool,
    tta: bool,
    error_reference: str,
    output_path: Path,
) -> None:
    arrays = load_drive_sample(sample)
    image = prepare_display_image(arrays["image"])
    manual_1 = binarize_mask(arrays["manual_1"])
    manual_2 = binarize_mask(arrays["manual_2"]) if arrays["manual_2"] is not None else None

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
    fov = binarize_mask(arrays["fov_mask"]) if apply_fov else None
    prediction = remove_small_components(
        prediction,
        min_size=postprocess_min_size,
        fov_mask=fov,
    )

    reference = manual_1 if error_reference == "manual_1" else manual_2
    if reference is None:
        raise ValueError(f"Sample {sample.sample_id} does not have {error_reference}")

    false_positive = np.logical_and(prediction == 1, reference == 0)
    false_negative = np.logical_and(prediction == 0, reference == 1)
    dice_1 = dice_score(manual_1, prediction)
    dice_2 = dice_score(manual_2, prediction) if manual_2 is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    title = f"Sample {sample.sample_id} - patch {patch_size} stride {stride} - threshold {threshold:.2f}"
    if tta:
        title += " - TTA"
    if dice_2 is None:
        title += f" - DICE exp1 {dice_1:.3f}"
    else:
        title += f" - DICE exp1 {dice_1:.3f} / exp2 {dice_2:.3f}"
    fig.suptitle(title, fontsize=14)

    show_rgb(axes[0, 0], image, "Imagen original")
    show_mask(axes[0, 1], manual_1, "Mascara experto 1")
    show_mask(axes[0, 2], manual_2, "Mascara experto 2")
    show_mask(axes[1, 0], prediction, "Prediccion patch ensemble")
    show_rgb(
        axes[1, 1],
        overlay_error_mask(image, false_positive, color=(0.0, 1.0, 1.0)),
        "Falsos positivos (cian)",
    )
    show_rgb(
        axes[1, 2],
        overlay_error_mask(image, false_negative, color=(1.0, 0.0, 1.0)),
        "Falsos negativos (magenta)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def overlay_error_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = 1.0,
) -> np.ndarray:
    overlay = image.copy()
    mask = np.asarray(mask).astype(bool)
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_array
    return np.clip(overlay, 0.0, 1.0)


def show_rgb(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def show_mask(axis, mask: np.ndarray | None, title: str) -> None:
    if mask is None:
        axis.text(0.5, 0.5, "No disponible", ha="center", va="center")
    else:
        axis.imshow((mask > 0).astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true = (np.asarray(y_true) > 0).astype(np.float32).ravel()
    y_pred = (np.asarray(y_pred) > 0).astype(np.float32).ravel()
    intersection = float(np.sum(y_true * y_pred))
    denominator = float(np.sum(y_true) + np.sum(y_pred))
    return (2.0 * intersection + smooth) / (denominator + smooth)


if __name__ == "__main__":
    main()
