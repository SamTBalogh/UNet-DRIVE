from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    samples = select_samples(
        list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test"),
        sample_ids=parse_sample_ids(args.sample_ids),
        max_samples=args.max_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(models)} models: {', '.join(path.name for path in model_paths)}")
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    print(f"Generating diagnostics for {len(samples)} samples")

    for sample in samples:
        output_path = output_dir / f"{sample.sample_id}_{args.split}_diagnostic.png"
        create_diagnostic_figure(
            models=models,
            sample=sample,
            image_size=image_size,
            resize_strategy=resize_strategy,
            threshold=args.threshold,
            apply_fov=args.apply_fov,
            error_reference=args.error_reference,
            error_dilation=args.error_dilation,
            output_path=output_path,
        )
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual diagnostics for an ensemble prediction.")
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_flips_valloss"),
        help="Directory containing fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to diagnose.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "figures" / "ensemble_diagnostics"),
        help="Where diagnostic figures are saved.",
    )
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
        "--error-dilation",
        type=int,
        default=0,
        help="Visual-only pixel dilation applied to error masks so thin vessels stand out.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during diagnostics.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help="Set TF_GPU_ALLOCATOR=cuda_malloc_async when using GPU.",
    )
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
    image_size: tuple[int, int],
    resize_strategy: str,
    threshold: float,
    apply_fov: bool,
    error_reference: str,
    error_dilation: int,
    output_path: Path,
) -> None:
    arrays = load_drive_sample(sample)
    image = prepare_display_image(arrays["image"])
    manual_1 = binarize_mask(arrays["manual_1"])
    manual_2 = binarize_mask(arrays["manual_2"]) if arrays["manual_2"] is not None else None

    probability = predict_ensemble_probability(
        models=models,
        sample=sample,
        image_size=image_size,
        resize_strategy=resize_strategy,
        apply_fov=apply_fov,
    )
    prediction = probability_to_binary_mask(probability, threshold=threshold)

    reference = manual_1 if error_reference == "manual_1" else manual_2
    if reference is None:
        raise ValueError(f"Sample {sample.sample_id} does not have {error_reference}")

    false_positive = dilate_binary_mask(np.logical_and(prediction == 1, reference == 0), error_dilation)
    false_negative = dilate_binary_mask(np.logical_and(prediction == 0, reference == 1), error_dilation)
    dice_1 = dice_score(manual_1, prediction)
    dice_2 = dice_score(manual_2, prediction) if manual_2 is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    title = f"Sample {sample.sample_id} - threshold {threshold:.2f}"
    if dice_2 is None:
        title += f" - DICE exp1 {dice_1:.3f}"
    else:
        title += f" - DICE exp1 {dice_1:.3f} / exp2 {dice_2:.3f}"
    fig.suptitle(title, fontsize=14)

    show_rgb(axes[0, 0], image, "Imagen original")
    show_mask(axes[0, 1], manual_1, "Mascara experto 1")
    show_mask(axes[0, 2], manual_2, "Mascara experto 2")
    show_mask(axes[1, 0], prediction, "Prediccion ensemble")
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


def dilate_binary_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
        expanded = np.zeros_like(mask, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                expanded |= padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
        mask = expanded
    return mask


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
