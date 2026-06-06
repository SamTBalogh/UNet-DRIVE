from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR, OUTPUTS_DIR
from data import list_drive_samples
from ensemble import configure_inference_environment
from patch_inference import (
    load_patch_metadata,
    load_patch_models,
    predict_patch_ensemble_probability,
    probability_to_binary_mask,
    resolve_patch_size,
    resolve_stride,
    save_binary_png,
)


def main() -> None:
    args = parse_args()
    configure_inference_environment(device=args.device, cuda_malloc_async=args.cuda_malloc_async)

    models, model_paths = load_patch_models(args.models_dir, pattern=args.model_pattern)
    metadata = load_patch_metadata(args.models_dir)
    patch_size = resolve_patch_size(models, metadata=metadata, fallback=args.patch_size)
    stride = resolve_stride(args.stride, patch_size=patch_size)

    samples = select_samples(
        list_drive_samples(args.data_dir, split=args.split, require_manual_2=False),
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
    print(f"Generating {len(samples)} masks")

    for sample in samples:
        probability = predict_patch_ensemble_probability(
            models=models,
            sample=sample,
            patch_size=patch_size,
            stride=stride,
            batch_size=args.predict_batch_size,
            apply_fov=args.apply_fov,
            tta=args.tta,
        )
        mask = probability_to_binary_mask(probability, threshold=args.threshold)
        output_path = output_dir / f"{sample.sample_id}_{args.split}_segmentation.png"
        save_binary_png(mask, output_path)
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG vessel segmentations with a patch-trained ensemble."
    )
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_patch128_balanced"),
        help="Directory containing patch-trained fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to predict.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "segmentations" / "patch_ensemble_test"),
        help="Where PNG masks are saved.",
    )
    parser.add_argument("--patch-size", type=int, default=128, help="Fallback patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride. Defaults to patch_size / 2.")
    parser.add_argument("--predict-batch-size", type=int, default=16, help="Patch prediction batch size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for vessel pixels.")
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
        help="Default is cpu to avoid GPU memory pressure during prediction.",
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


if __name__ == "__main__":
    main()
