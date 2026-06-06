from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import keras
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from augment import (
    apply_geometric_transform,
    apply_photometric_transform,
    sample_transform_params,
)
from config import (
    DATA_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_FOLDS,
    DEFAULT_LEARNING_RATE,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from metrics import (
    bce_dice_loss,
    dice_coef,
    dice_loss,
    focal_tversky_loss,
    thin_weighted_dice_loss,
    weighted_bce_dice_loss,
)
from model import build_unet
from patches import (
    PATCH_CATEGORIES,
    PatchSamplingConfig,
    PatchSource,
    build_patch_sources,
    sample_balanced_patch_arrays,
    sample_balanced_patch_batch,
    summarize_patch_sources,
    write_patch_source_summary,
)


PATCH_BATCH_SIZE = 8


def main() -> None:
    args = parse_args()
    validate_args(args)
    keras.utils.set_random_seed(args.seed)

    samples = list_training_samples(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = patch_sampling_config(args)
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Patch categories: {config.category_fractions()}")
    print(f"Training images: {len(samples)}")

    if args.folds <= 1:
        train_single_split(samples, output_dir, config, args)
    else:
        train_cross_validation(samples, output_dir, config, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net on balanced DRIVE image patches.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "models" / "cv_bce_dice_patch128_balanced"),
        help="Where to save patch-trained models and histories.",
    )
    parser.add_argument("--patch-size", type=int, default=128, help="Square patch size used as model input.")
    parser.add_argument(
        "--candidate-stride",
        type=int,
        default=16,
        help="Stride used to precompute candidate patch top-left positions.",
    )
    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=512,
        help="Approximate number of sampled training patches per source image and epoch.",
    )
    parser.add_argument(
        "--val-patches-per-image",
        type=int,
        default=256,
        help="Number of deterministic validation patches sampled per validation image.",
    )
    parser.add_argument("--batch-size", type=int, default=PATCH_BATCH_SIZE, help="Patch training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--loss",
        choices=(
            "bce",
            "dice",
            "bce_dice",
            "weighted_bce_dice",
            "focal_tversky",
            "thin_weighted_dice",
        ),
        default="bce_dice",
        help="Training loss. Thin-vessel experiments should use weighted_bce_dice, focal_tversky, or thin_weighted_dice.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Use 5 for cross-validation, 1 for a split.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction when --folds 1.")
    parser.add_argument("--base-filters", type=int, default=16, help="Initial number of U-Net filters.")
    parser.add_argument("--depth", type=int, default=4, help="Number of U-Net downsampling levels.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout after convolution blocks.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for pilot runs.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Build sources/model and sample patches without training.")
    parser.add_argument(
        "--min-fov-ratio",
        type=float,
        default=0.50,
        help="Minimum fraction of a patch that must lie inside the FoV.",
    )
    parser.add_argument(
        "--positive-ratio-min",
        type=float,
        default=0.01,
        help="Minimum vessel ratio for a patch to be a positive candidate.",
    )
    parser.add_argument(
        "--background-ratio-max",
        type=float,
        default=0.002,
        help="Maximum vessel ratio for a patch to be a background candidate.",
    )
    parser.add_argument(
        "--thin-ratio-min",
        type=float,
        default=0.001,
        help="Minimum thin-vessel ratio for a patch to be a thin/difficult candidate.",
    )
    parser.add_argument(
        "--thin-neighbor-threshold",
        type=int,
        default=4,
        help="3x3 vessel-neighbor threshold used to approximate thin vessels.",
    )
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=0.60,
        help="Fraction of each batch sampled from positive patches.",
    )
    parser.add_argument(
        "--thin-fraction",
        type=float,
        default=0.20,
        help="Fraction of each batch sampled from thin/difficult patches.",
    )
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.20,
        help="Fraction of each batch sampled from background patches.",
    )
    parser.add_argument("--augment-flips", action="store_true", help="Apply random synchronized flips to patches.")
    parser.add_argument(
        "--augment-rich",
        action="store_true",
        help="Apply random geometric and photometric augmentation to patches.",
    )
    parser.add_argument("--augment-rotation-degrees", type=float, default=12.0)
    parser.add_argument("--augment-shift-pixels", type=int, default=16)
    parser.add_argument("--augment-brightness", type=float, default=0.15)
    parser.add_argument("--augment-contrast", type=float, default=0.15)
    parser.add_argument("--augment-gamma", type=float, default=0.15)
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--checkpoint-monitor", default="val_loss", help="Metric monitored by ModelCheckpoint.")
    parser.add_argument("--checkpoint-mode", choices=("min", "max", "auto"), default="min")
    parser.add_argument("--early-stopping-monitor", default="val_loss", help="Metric monitored by EarlyStopping.")
    parser.add_argument("--early-stopping-mode", choices=("min", "max", "auto"), default="min")
    parser.add_argument("--patience", type=int, default=16, help="EarlyStopping patience.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.patch_size % (2**args.depth) != 0:
        raise ValueError(f"--patch-size must be divisible by {2**args.depth} for depth={args.depth}.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.patches_per_image <= 0:
        raise ValueError("--patches-per-image must be positive.")
    if args.val_patches_per_image <= 0:
        raise ValueError("--val-patches-per-image must be positive.")


def list_training_samples(args: argparse.Namespace):
    from data import list_drive_samples

    samples = list_drive_samples(args.data_dir, split="training", require_manual_2=False)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if len(samples) < 2:
        raise RuntimeError("At least two training samples are required.")
    return samples


def patch_sampling_config(args: argparse.Namespace) -> PatchSamplingConfig:
    return PatchSamplingConfig(
        patch_size=args.patch_size,
        candidate_stride=args.candidate_stride,
        min_fov_ratio=args.min_fov_ratio,
        positive_ratio_min=args.positive_ratio_min,
        background_ratio_max=args.background_ratio_max,
        thin_ratio_min=args.thin_ratio_min,
        thin_neighbor_threshold=args.thin_neighbor_threshold,
        positive_fraction=args.positive_fraction,
        thin_fraction=args.thin_fraction,
        background_fraction=args.background_fraction,
    )


def train_single_split(
    samples: list,
    output_dir: Path,
    config: PatchSamplingConfig,
    args: argparse.Namespace,
) -> None:
    indices = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_one_fold(samples, train_idx, val_idx, output_dir, config, args, fold_name="single_split")


def train_cross_validation(
    samples: list,
    output_dir: Path,
    config: PatchSamplingConfig,
    args: argparse.Namespace,
) -> None:
    if args.folds > len(samples):
        raise ValueError(f"Cannot use {args.folds} folds with only {len(samples)} samples.")

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(samples), start=1):
        train_one_fold(
            samples,
            train_idx,
            val_idx,
            output_dir,
            config,
            args,
            fold_name=f"fold_{fold_index}",
        )


def train_one_fold(
    samples: list,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    config: PatchSamplingConfig,
    args: argparse.Namespace,
    fold_name: str,
) -> None:
    print(f"\nPreparing {fold_name}: {len(train_idx)} train images / {len(val_idx)} validation images")
    train_samples = [samples[index] for index in train_idx]
    val_samples = [samples[index] for index in val_idx]
    train_sources = build_patch_sources(train_samples, config=config)
    val_sources = build_patch_sources(val_samples, config=config)

    write_patch_source_summary(train_sources, output_dir / f"{fold_name}_train_patch_candidates.csv")
    write_patch_source_summary(val_sources, output_dir / f"{fold_name}_val_patch_candidates.csv")
    print_patch_summary("train", train_sources)
    print_patch_summary("validation", val_sources)

    steps_per_epoch = int(np.ceil(len(train_samples) * args.patches_per_image / args.batch_size))
    val_patch_count = len(val_samples) * args.val_patches_per_image
    val_x, val_y = sample_balanced_patch_arrays(
        sources=val_sources,
        total_patches=val_patch_count,
        batch_size=args.batch_size,
        rng=np.random.default_rng(fold_seed(args.seed, fold_name) + 100_000),
        category_fractions=config.category_fractions(),
    )
    print(f"Validation patches: X={val_x.shape} y={val_y.shape}")

    model = compile_model(args.patch_size, args)
    if args.dry_run:
        train_batch = BalancedPatchDataset(
            sources=train_sources,
            batch_size=args.batch_size,
            steps_per_epoch=1,
            config=config,
            seed=fold_seed(args.seed, fold_name),
            args=args,
            augment=True,
        )[0]
        print(f"Dry-run train batch: X={train_batch[0].shape} y={train_batch[1].shape}")
        print(f"Dry-run validation range: X {val_x.min():.3f}-{val_x.max():.3f}; masks {np.unique(val_y).tolist()}")
        model.summary()
        save_fold_metadata(
            output_dir / f"{fold_name}_metadata.json",
            train_idx=train_idx,
            val_idx=val_idx,
            train_samples=train_samples,
            val_samples=val_samples,
            steps_per_epoch=steps_per_epoch,
            val_patch_count=val_patch_count,
            args=args,
            config=config,
        )
        return

    train_dataset = BalancedPatchDataset(
        sources=train_sources,
        batch_size=args.batch_size,
        steps_per_epoch=steps_per_epoch,
        config=config,
        seed=fold_seed(args.seed, fold_name),
        args=args,
        augment=True,
    )
    model_path = output_dir / f"{fold_name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=args.checkpoint_monitor,
            mode=args.checkpoint_mode,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=args.early_stopping_monitor,
            mode=args.early_stopping_mode,
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    print(f"Training steps per epoch: {steps_per_epoch}")
    history = model.fit(
        train_dataset,
        validation_data=(val_x, val_y),
        validation_batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not model_path.exists():
        model.save(model_path)
    save_history(history.history, output_dir / f"{fold_name}_history.csv")
    save_fold_metadata(
        output_dir / f"{fold_name}_metadata.json",
        train_idx=train_idx,
        val_idx=val_idx,
        train_samples=train_samples,
        val_samples=val_samples,
        steps_per_epoch=steps_per_epoch,
        val_patch_count=val_patch_count,
        args=args,
        config=config,
    )


class BalancedPatchDataset(keras.utils.PyDataset):
    def __init__(
        self,
        sources: list[PatchSource],
        batch_size: int,
        steps_per_epoch: int,
        config: PatchSamplingConfig,
        seed: int,
        args: argparse.Namespace,
        augment: bool,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.config = config
        self.seed = seed
        self.args = args
        self.augment = augment
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * self.steps_per_epoch + index)
        x, y, _metadata = sample_balanced_patch_batch(
            sources=self.sources,
            batch_size=self.batch_size,
            rng=rng,
            category_fractions=self.config.category_fractions(),
        )
        if self.augment:
            x, y = augment_patch_batch(x, y, self.args, rng)
        return x, y

    def on_epoch_end(self) -> None:
        self.epoch += 1


def augment_patch_batch(
    images: np.ndarray,
    masks: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    augmented_images = []
    augmented_masks = []
    for image, mask in zip(images, masks):
        aug_image = image
        aug_mask = mask

        if args.augment_flips:
            if rng.random() < 0.5:
                aug_image = np.flip(aug_image, axis=1)
                aug_mask = np.flip(aug_mask, axis=1)
            if rng.random() < 0.5:
                aug_image = np.flip(aug_image, axis=0)
                aug_mask = np.flip(aug_mask, axis=0)
            aug_image = np.ascontiguousarray(aug_image)
            aug_mask = np.ascontiguousarray(aug_mask)

        if args.augment_rich:
            params = sample_transform_params(
                rng=rng,
                rotation_degrees=args.augment_rotation_degrees,
                shift_pixels=args.augment_shift_pixels,
                brightness_delta=args.augment_brightness,
                contrast_delta=args.augment_contrast,
                gamma_delta=args.augment_gamma,
                noise_std=args.augment_noise_std,
            )
            aug_image = apply_geometric_transform(
                aug_image,
                angle=params["angle"],
                shift_y=params["shift_y"],
                shift_x=params["shift_x"],
                is_mask=False,
            )
            aug_mask = apply_geometric_transform(
                aug_mask,
                angle=params["angle"],
                shift_y=params["shift_y"],
                shift_x=params["shift_x"],
                is_mask=True,
            )
            aug_image = apply_photometric_transform(
                aug_image,
                rng=rng,
                brightness=params["brightness"],
                contrast=params["contrast"],
                gamma=params["gamma"],
                noise_std=params["noise_std"],
            )

        augmented_images.append(np.ascontiguousarray(aug_image).astype(np.float32))
        augmented_masks.append((np.ascontiguousarray(aug_mask) > 0.5).astype(np.float32))

    return np.stack(augmented_images).astype(np.float32), np.stack(augmented_masks).astype(np.float32)


def compile_model(patch_size: int, args: argparse.Namespace) -> keras.Model:
    model = build_unet(
        input_shape=(patch_size, patch_size, 3),
        base_filters=args.base_filters,
        depth=args.depth,
        dropout=args.dropout,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=select_loss(args.loss),
        metrics=[dice_coef],
    )
    return model


def select_loss(loss_name: str):
    if loss_name == "bce":
        return "binary_crossentropy"
    if loss_name == "dice":
        return dice_loss
    if loss_name == "bce_dice":
        return bce_dice_loss
    if loss_name == "weighted_bce_dice":
        return weighted_bce_dice_loss
    if loss_name == "focal_tversky":
        return focal_tversky_loss
    if loss_name == "thin_weighted_dice":
        return thin_weighted_dice_loss
    raise ValueError(f"Unsupported loss: {loss_name}")


def print_patch_summary(label: str, sources: list[PatchSource]) -> None:
    rows = summarize_patch_sources(sources)
    totals = {
        category: sum(int(row[f"{category}_candidates"]) for row in rows)
        for category in PATCH_CATEGORIES
    }
    total_all = sum(int(row["all_candidates"]) for row in rows)
    print(
        f"{label.capitalize()} candidates: "
        f"positive={totals['positive']} "
        f"thin={totals['thin']} "
        f"background={totals['background']} "
        f"all={total_all}"
    )


def save_history(history: dict[str, list[float]], output_path: Path) -> None:
    keys = list(history.keys())
    rows = zip(*[history[key] for key in keys])
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for epoch, values in enumerate(rows, start=1):
            writer.writerow([epoch, *values])


def save_fold_metadata(
    output_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_samples: list,
    val_samples: list,
    steps_per_epoch: int,
    val_patch_count: int,
    args: argparse.Namespace,
    config: PatchSamplingConfig,
) -> None:
    metadata = {
        "training_unit": "balanced_patches",
        "requires_sliding_window_inference": True,
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "train_sample_ids": [sample.sample_id for sample in train_samples],
        "validation_sample_ids": [sample.sample_id for sample in val_samples],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "image_size": [args.patch_size, args.patch_size],
        "patch_size": args.patch_size,
        "candidate_stride": args.candidate_stride,
        "patches_per_image": args.patches_per_image,
        "val_patches_per_image": args.val_patches_per_image,
        "steps_per_epoch": steps_per_epoch,
        "validation_patch_count": val_patch_count,
        "min_fov_ratio": config.min_fov_ratio,
        "positive_ratio_min": config.positive_ratio_min,
        "background_ratio_max": config.background_ratio_max,
        "thin_ratio_min": config.thin_ratio_min,
        "thin_neighbor_threshold": config.thin_neighbor_threshold,
        "category_fractions": config.category_fractions(),
        "base_filters": args.base_filters,
        "depth": args.depth,
        "dropout": args.dropout,
        "augment_flips": args.augment_flips,
        "augment_rich": args.augment_rich,
        "augment_rotation_degrees": args.augment_rotation_degrees,
        "augment_shift_pixels": args.augment_shift_pixels,
        "augment_brightness": args.augment_brightness,
        "augment_contrast": args.augment_contrast,
        "augment_gamma": args.augment_gamma,
        "augment_noise_std": args.augment_noise_std,
        "checkpoint_monitor": args.checkpoint_monitor,
        "checkpoint_mode": args.checkpoint_mode,
        "early_stopping_monitor": args.early_stopping_monitor,
        "early_stopping_mode": args.early_stopping_mode,
        "patience": args.patience,
        "seed": args.seed,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def fold_seed(seed: int, fold_name: str) -> int:
    return seed + sum(ord(char) for char in fold_name)


if __name__ == "__main__":
    main()
