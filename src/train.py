from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import keras
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from augment import add_flip_augmentations
from config import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FOLDS,
    DEFAULT_LEARNING_RATE,
    IMAGE_SIZE,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from data import load_drive_arrays, list_drive_samples, resolve_preprocessed_image_size
from metrics import bce_dice_loss, dice_coef, dice_loss
from model import build_unet


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    samples = list_drive_samples(args.data_dir, split="training", require_manual_2=False)
    if args.max_samples:
        samples = samples[: args.max_samples]

    if len(samples) < 2:
        raise RuntimeError("At least two training samples are required.")

    args.pad_multiple = max(args.pad_multiple, 2**args.depth)
    image_size = resolve_preprocessed_image_size(
        samples,
        resize_strategy=args.resize_strategy,
        image_size=(args.image_height, args.image_width),
        pad_multiple=args.pad_multiple,
    )
    args.resolved_image_size = image_size
    print(f"Preprocessing strategy: {args.resize_strategy}")
    print(f"Model image size: {image_size}")

    x, y = load_drive_arrays(
        samples,
        image_size=image_size,
        target="manual_1",
        resize_strategy=args.resize_strategy,
    )
    print(f"Loaded X={x.shape} y={y.shape}")
    print(f"Image range: {x.min():.3f} to {x.max():.3f}")
    print(f"Mask values: {np.unique(y).tolist()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        model = compile_model(image_size, args)
        model.summary()
        return

    if args.folds <= 1:
        train_single_split(x, y, output_dir, image_size, args)
    else:
        train_cross_validation(x, y, output_dir, image_size, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net on DRIVE training images.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "models"), help="Where to save models and histories.")
    parser.add_argument("--image-height", type=int, default=IMAGE_SIZE[0], help="Model input height.")
    parser.add_argument("--image-width", type=int, default=IMAGE_SIZE[1], help="Model input width.")
    parser.add_argument(
        "--resize-strategy",
        choices=("resize", "pad"),
        default="resize",
        help="Preprocessing strategy. 'pad' keeps aspect ratio and pads to --pad-multiple.",
    )
    parser.add_argument("--pad-multiple", type=int, default=16, help="Pad image dimensions to this multiple.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--loss",
        choices=("bce", "dice", "bce_dice"),
        default="bce",
        help="Training loss. Use bce_dice for the next real experiment.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Use 5 for cross-validation, 1 for a simple split.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction when --folds 1.")
    parser.add_argument("--base-filters", type=int, default=16, help="Initial number of U-Net filters.")
    parser.add_argument("--depth", type=int, default=4, help="Number of U-Net downsampling levels.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout after convolution blocks.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a pilot run.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and build the model without training.")
    parser.add_argument("--augment-flips", action="store_true", help="Add synchronized flip augmentations to training folds.")
    parser.add_argument("--checkpoint-monitor", default="val_dice_coef", help="Metric monitored by ModelCheckpoint.")
    parser.add_argument("--checkpoint-mode", choices=("min", "max", "auto"), default="max", help="ModelCheckpoint mode.")
    parser.add_argument("--early-stopping-monitor", default="val_dice_coef", help="Metric monitored by EarlyStopping.")
    parser.add_argument("--early-stopping-mode", choices=("min", "max", "auto"), default="max", help="EarlyStopping mode.")
    parser.add_argument("--patience", type=int, default=8, help="EarlyStopping patience.")
    return parser.parse_args()


def compile_model(image_size: tuple[int, int], args: argparse.Namespace) -> keras.Model:
    model = build_unet(
        input_shape=(image_size[0], image_size[1], 3),
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
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_single_split(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    train_idx, val_idx = train_test_split(
        np.arange(len(x)),
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_one_fold(x, y, train_idx, val_idx, output_dir, image_size, args, fold_name="single_split")


def train_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    if args.folds > len(x):
        raise ValueError(f"Cannot use {args.folds} folds with only {len(x)} samples.")

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(x), start=1):
        train_one_fold(
            x,
            y,
            train_idx,
            val_idx,
            output_dir,
            image_size,
            args,
            fold_name=f"fold_{fold_index}",
        )


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    image_size: tuple[int, int],
    args: argparse.Namespace,
    fold_name: str,
) -> None:
    print(f"\nTraining {fold_name}: {len(train_idx)} train / {len(val_idx)} validation")
    model = compile_model(image_size, args)

    x_train = x[train_idx]
    y_train = y[train_idx]
    if args.augment_flips:
        x_train, y_train = add_flip_augmentations(x_train, y_train)
        print(f"After flip augmentation: {len(x_train)} training samples")

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

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
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
        args=args,
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
    args: argparse.Namespace,
) -> None:
    metadata = {
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "image_size": list(args.resolved_image_size),
        "resize_strategy": args.resize_strategy,
        "pad_multiple": args.pad_multiple,
        "base_filters": args.base_filters,
        "depth": args.depth,
        "dropout": args.dropout,
        "augment_flips": args.augment_flips,
        "checkpoint_monitor": args.checkpoint_monitor,
        "checkpoint_mode": args.checkpoint_mode,
        "early_stopping_monitor": args.early_stopping_monitor,
        "early_stopping_mode": args.early_stopping_mode,
        "patience": args.patience,
        "seed": args.seed,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
