from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data import binarize_mask, list_drive_samples, load_drive_sample


def main() -> None:
    args = parse_args()
    samples = list_drive_samples(args.data_dir, split=args.split)

    if not samples:
        raise RuntimeError(f"No DRIVE samples found in {args.data_dir} for split {args.split}")

    selected = samples[: args.num_samples]
    print(f"Found {len(samples)} samples in split '{args.split}'.")
    print("First paired samples:")
    for sample in selected:
        print(
            f"- id={sample.sample_id} | image={sample.image_path.name} | "
            f"manual_1={sample.manual_1_path.name} | "
            f"manual_2={sample.manual_2_path.name if sample.manual_2_path else '-'} | "
            f"fov={sample.fov_mask_path.name}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_samples_figure(selected, output_path)
    print(f"Saved visual check to: {output_path}")

    if args.show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List paired DRIVE images/manual masks and save a visual check figure."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/DRIVE",
        help="Path to the DRIVE dataset root.",
    )
    parser.add_argument(
        "--split",
        choices=("training", "test"),
        default="training",
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/drive_samples_check.png",
        help="Path where the visual check figure will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the matplotlib window after saving the figure.",
    )
    return parser.parse_args()


def save_samples_figure(samples, output_path: Path) -> None:
    has_manual_2 = any(sample.manual_2_path is not None for sample in samples)
    columns = 4 if has_manual_2 else 3
    column_titles = ["Image", "Manual 1", "FoV mask"]
    if has_manual_2:
        column_titles.insert(2, "Manual 2")

    figure, axes = plt.subplots(
        nrows=len(samples),
        ncols=columns,
        figsize=(4 * columns, 3.2 * len(samples)),
        squeeze=False,
    )

    for row, sample in enumerate(samples):
        arrays = load_drive_sample(sample)
        manual_1 = binarize_mask(arrays["manual_1"])
        fov_mask = binarize_mask(arrays["fov_mask"])
        manual_2 = (
            binarize_mask(arrays["manual_2"])
            if arrays["manual_2"] is not None
            else None
        )

        row_images = [arrays["image"], manual_1]
        if has_manual_2:
            row_images.append(manual_2)
        row_images.append(fov_mask)

        for col, image in enumerate(row_images):
            ax = axes[row][col]
            if image is None:
                ax.text(0.5, 0.5, "No manual 2", ha="center", va="center")
            else:
                cmap = None if col == 0 else "gray"
                ax.imshow(image, cmap=cmap)
            ax.set_title(f"{sample.sample_id} - {column_titles[col]}")
            ax.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
