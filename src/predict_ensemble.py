from __future__ import annotations

import argparse
from pathlib import Path

from config import DATA_DIR, IMAGE_SIZE, OUTPUTS_DIR
from data import list_drive_samples
from ensemble import (
    configure_inference_environment,
    find_model_paths,
    infer_resize_strategy,
    load_models,
    predict_ensemble_probability,
    probability_to_binary_mask,
    resolve_models_image_size,
    save_binary_png,
)


def main() -> None:
    args = parse_args()
    configure_inference_environment(device=args.device, cuda_malloc_async=args.cuda_malloc_async)

    resize_strategy = infer_resize_strategy(args.models_dir, requested=args.resize_strategy)
    model_paths = find_model_paths(args.models_dir, pattern=args.model_pattern)
    models = load_models(model_paths)
    image_size = resolve_models_image_size(models, fallback=(args.image_height, args.image_width))

    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(models)} models: {', '.join(path.name for path in model_paths)}")
    print(f"Preprocessing strategy: {resize_strategy}")
    print(f"Model image size: {image_size}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    for sample in samples:
        probability = predict_ensemble_probability(
            models=models,
            sample=sample,
            image_size=image_size,
            resize_strategy=resize_strategy,
            apply_fov=args.apply_fov,
            tta=args.tta,
        )
        mask = probability_to_binary_mask(probability, threshold=args.threshold)
        output_path = output_dir / f"{sample.sample_id}_{args.split}_segmentation.png"
        save_binary_png(mask, output_path)
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG vessel segmentations with a model ensemble.")
    parser.add_argument(
        "--models-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_flips_valloss"),
        help="Directory containing fold .keras models.",
    )
    parser.add_argument("--model-pattern", default="fold_*.keras", help="Glob pattern for model files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to predict.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "segmentations" / "ensemble_test"),
        help="Where PNG masks are saved.",
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
        "--tta",
        action="store_true",
        help="Average original, horizontal flip, vertical flip and both-flip predictions before thresholding.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Default is cpu to avoid GPU memory pressure during prediction.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help="Set TF_GPU_ALLOCATOR=cuda_malloc_async when using GPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
