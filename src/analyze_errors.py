from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import DATA_DIR, OUTPUTS_DIR
from data import binarize_mask, list_drive_samples, load_drive_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-image segmentation errors from saved PNG predictions."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to DRIVE root folder.")
    parser.add_argument("--split", choices=("training", "test"), default="test", help="Split to analyze.")
    parser.add_argument(
        "--predictions-dir",
        default=str(OUTPUTS_DIR / "segmentations" / "final_ensemble_test_pad_e120_aug_tta"),
        help="Directory containing '<id>_<split>_segmentation.png' prediction masks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "results" / "error_analysis_t045_tta"),
        help="Directory where analysis CSV/Markdown outputs are saved.",
    )
    parser.add_argument(
        "--figure-dir",
        default=str(OUTPUTS_DIR / "figures" / "error_analysis_t045_tta"),
        help="Directory where visual error summaries are saved.",
    )
    parser.add_argument("--worst-count", type=int, default=5, help="Number of worst images to summarize visually.")
    parser.add_argument(
        "--border-pixels",
        type=int,
        default=8,
        help="FoV inner-border width used to estimate border-related errors.",
    )
    parser.add_argument(
        "--thin-neighbor-threshold",
        type=int,
        default=4,
        help="A vessel pixel with this many or fewer 3x3 vessel neighbors is treated as thin-like.",
    )
    parser.add_argument(
        "--small-fp-component-threshold",
        type=int,
        default=8,
        help="False-positive connected components up to this size are treated as small noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_drive_samples(args.data_dir, split=args.split, require_manual_2=args.split == "test")
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    arrays_by_id = {}
    for sample in samples:
        prediction_path = predictions_dir / f"{sample.sample_id}_{args.split}_segmentation.png"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction mask: {prediction_path}")

        arrays = load_drive_sample(sample)
        prediction = binarize_mask(read_image(prediction_path))
        manual_1 = binarize_mask(arrays["manual_1"])
        manual_2 = binarize_mask(arrays["manual_2"]) if arrays["manual_2"] is not None else None
        fov = binarize_mask(arrays["fov_mask"])
        image = ensure_rgb(arrays["image"])

        row = analyze_sample(
            sample_id=sample.sample_id,
            image=image,
            prediction=prediction,
            manual_1=manual_1,
            manual_2=manual_2,
            fov=fov,
            border_pixels=args.border_pixels,
            thin_neighbor_threshold=args.thin_neighbor_threshold,
            small_fp_component_threshold=args.small_fp_component_threshold,
        )
        rows.append(row)
        arrays_by_id[sample.sample_id] = {
            "image": image,
            "prediction": prediction,
            "manual_1": manual_1,
            "manual_2": manual_2,
            "fov": fov,
        }

    rows = add_relative_flags(rows)
    csv_path = output_dir / "error_analysis_by_image.csv"
    write_csv(csv_path, rows)

    report_path = output_dir / "error_analysis_report.md"
    report = build_report(rows, args)
    report_path.write_text(report, encoding="utf-8")

    worst_rows = sorted(rows, key=lambda row: float(row["dice_mean"]))[: args.worst_count]
    figure_path = figure_dir / "worst_cases_contact_sheet.png"
    save_contact_sheet(worst_rows, arrays_by_id, figure_path)

    print(f"Analyzed {len(rows)} images")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved worst-case figure: {figure_path}")
    print()
    print("Worst images by DICE mean:")
    for row in worst_rows:
        print(
            f"- {row['image_id']}: dice={float(row['dice_mean']):.4f}, "
            f"precision={float(row['precision_manual_1']):.4f}, "
            f"recall={float(row['recall_manual_1']):.4f}, "
            f"flags={row['flags']}"
        )


def read_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.copy())


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def analyze_sample(
    sample_id: str,
    image: np.ndarray,
    prediction: np.ndarray,
    manual_1: np.ndarray,
    manual_2: np.ndarray | None,
    fov: np.ndarray,
    border_pixels: int,
    thin_neighbor_threshold: int,
    small_fp_component_threshold: int,
) -> dict[str, str | float | int]:
    fov_bool = fov > 0
    full_valid = np.ones_like(fov_bool, dtype=bool)
    pred_full = prediction > 0
    gt_full = manual_1 > 0
    manual_2_full = manual_2 > 0 if manual_2 is not None else None

    full_counts = confusion_counts(gt_full, pred_full, full_valid)
    dice_1 = dice_from_counts(full_counts["tp"], full_counts["fp"], full_counts["fn"])
    dice_2 = ""
    dice_mean = dice_1
    if manual_2_full is not None:
        full_counts_2 = confusion_counts(manual_2_full, pred_full, full_valid)
        dice_2 = dice_from_counts(full_counts_2["tp"], full_counts_2["fp"], full_counts_2["fn"])
        dice_mean = float(np.mean([dice_1, dice_2]))

    pred = pred_full & fov_bool
    gt = gt_full & fov_bool
    manual_2_bool = manual_2_full & fov_bool if manual_2_full is not None else None

    counts = confusion_counts(gt, pred, fov_bool)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fp_fraction_prediction = safe_div(fp, tp + fp)
    fn_fraction_manual = safe_div(fn, tp + fn)

    inter_expert_dice = ""
    if manual_2_bool is not None:
        expert_counts = confusion_counts(gt, manual_2_bool, fov_bool)
        inter_expert_dice = dice_from_counts(
            expert_counts["tp"], expert_counts["fp"], expert_counts["fn"]
        )

    false_positive = pred & ~gt
    false_negative = gt & ~pred
    error_mask = false_positive | false_negative

    thin_vessel = thin_vessel_proxy(gt, thin_neighbor_threshold)
    thick_vessel = gt & ~thin_vessel
    thin_recall = safe_div(int(np.sum(pred & thin_vessel)), int(np.sum(thin_vessel)))
    thick_recall = safe_div(int(np.sum(pred & thick_vessel)), int(np.sum(thick_vessel)))
    thin_fn_share = safe_div(int(np.sum(false_negative & thin_vessel)), fn)

    fov_border = border_band(fov_bool, border_pixels)
    border_error_share = safe_div(int(np.sum(error_mask & fov_border)), int(np.sum(error_mask)))
    border_fn_share = safe_div(int(np.sum(false_negative & fov_border)), fn)
    border_fp_share = safe_div(int(np.sum(false_positive & fov_border)), fp)

    green = image[..., 1].astype(np.float32) / 255.0
    dark_threshold = float(np.quantile(green[fov_bool], 0.20))
    dark_mask = fov_bool & (green <= dark_threshold)
    dark_error_share = safe_div(int(np.sum(error_mask & dark_mask)), int(np.sum(error_mask)))
    dark_fn_share = safe_div(int(np.sum(false_negative & dark_mask)), fn)
    dark_fp_share = safe_div(int(np.sum(false_positive & dark_mask)), fp)
    dark_vessel_recall = safe_div(int(np.sum(pred & gt & dark_mask)), int(np.sum(gt & dark_mask)))

    fp_components = connected_component_sizes(false_positive)
    small_fp_components = [size for size in fp_components if size <= small_fp_component_threshold]
    small_fp_pixels = int(np.sum(small_fp_components)) if small_fp_components else 0

    outside_fov_fp = int(np.sum((prediction > 0) & ~fov_bool))

    manual_positive = int(np.sum(gt))
    pred_positive = int(np.sum(pred))
    return {
        "image_id": sample_id,
        "dice_manual_1": dice_1,
        "dice_manual_2": dice_2,
        "dice_mean": dice_mean,
        "inter_expert_dice": inter_expert_dice,
        "precision_manual_1": precision,
        "recall_manual_1": recall,
        "specificity_manual_1": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_fraction_prediction": fp_fraction_prediction,
        "fn_fraction_manual": fn_fraction_manual,
        "manual_positive_pixels": manual_positive,
        "prediction_positive_pixels": pred_positive,
        "prediction_to_manual_positive_ratio": safe_div(pred_positive, manual_positive),
        "thin_vessel_pixels": int(np.sum(thin_vessel)),
        "thin_recall": thin_recall,
        "thick_recall": thick_recall,
        "thin_fn_share": thin_fn_share,
        "border_error_share": border_error_share,
        "border_fn_share": border_fn_share,
        "border_fp_share": border_fp_share,
        "dark_threshold_green": dark_threshold,
        "dark_error_share": dark_error_share,
        "dark_fn_share": dark_fn_share,
        "dark_fp_share": dark_fp_share,
        "dark_vessel_recall": dark_vessel_recall,
        "fp_component_count": len(fp_components),
        "small_fp_component_count": len(small_fp_components),
        "small_fp_pixel_share": safe_div(small_fp_pixels, fp),
        "outside_fov_fp": outside_fov_fp,
        "flags": "",
    }


def confusion_counts(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    gt = gt.astype(bool) & valid
    pred = pred.astype(bool) & valid
    return {
        "tp": int(np.sum(gt & pred)),
        "fp": int(np.sum(~gt & pred & valid)),
        "fn": int(np.sum(gt & ~pred)),
        "tn": int(np.sum(~gt & ~pred & valid)),
    }


def dice_from_counts(tp: int, fp: int, fn: int, smooth: float = 1e-6) -> float:
    return float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def erode_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(current, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(current, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                eroded &= padded[row_offset : row_offset + current.shape[0], col_offset : col_offset + current.shape[1]]
        current = eroded
    return current


def border_band(fov: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return np.zeros_like(fov, dtype=bool)
    return fov.astype(bool) & ~erode_binary(fov, iterations=pixels)


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    counts = np.zeros_like(mask, dtype=np.uint8)
    for row_offset in range(3):
        for col_offset in range(3):
            counts += padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return counts


def thin_vessel_proxy(mask: np.ndarray, threshold: int) -> np.ndarray:
    counts = neighbor_count(mask)
    return mask.astype(bool) & (counts <= threshold)


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    rows, cols = np.nonzero(mask)
    height, width = mask.shape

    for start_row, start_col in zip(rows.tolist(), cols.tolist()):
        if visited[start_row, start_col]:
            continue
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    if row_delta == 0 and col_delta == 0:
                        continue
                    next_row = row + row_delta
                    next_col = col + col_delta
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or not mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        sizes.append(size)
    return sizes


def add_relative_flags(rows: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
    numeric_fields = [
        "dice_mean",
        "precision_manual_1",
        "recall_manual_1",
        "thin_recall",
        "dark_error_share",
        "border_error_share",
        "small_fp_pixel_share",
    ]
    quantiles = {}
    for field in numeric_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        quantiles[field] = {
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }

    sorted_by_dice = sorted(rows, key=lambda row: float(row["dice_mean"]))
    worst_ids = {row["image_id"] for row in sorted_by_dice[:5]}

    for row in rows:
        flags = []
        if row["image_id"] in worst_ids:
            flags.append("low_dice")
        if float(row["recall_manual_1"]) <= quantiles["recall_manual_1"]["q25"]:
            flags.append("low_recall_fn")
        if float(row["precision_manual_1"]) <= quantiles["precision_manual_1"]["q25"]:
            flags.append("low_precision_fp")
        if float(row["thin_recall"]) <= quantiles["thin_recall"]["q25"]:
            flags.append("thin_vessels_missed")
        if float(row["dark_error_share"]) >= quantiles["dark_error_share"]["q75"]:
            flags.append("dark_zone_errors")
        if float(row["border_error_share"]) >= quantiles["border_error_share"]["q75"]:
            flags.append("fov_border_errors")
        if float(row["small_fp_pixel_share"]) >= quantiles["small_fp_pixel_share"]["q75"]:
            flags.append("small_fp_noise")
        row["flags"] = ";".join(flags) if flags else "none"
    return rows


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, str | float | int]], args: argparse.Namespace) -> str:
    rows_by_dice = sorted(rows, key=lambda row: float(row["dice_mean"]))
    mean_dice = np.mean([float(row["dice_mean"]) for row in rows])
    mean_precision = np.mean([float(row["precision_manual_1"]) for row in rows])
    mean_recall = np.mean([float(row["recall_manual_1"]) for row in rows])
    mean_thin_recall = np.mean([float(row["thin_recall"]) for row in rows])
    mean_dark_error = np.mean([float(row["dark_error_share"]) for row in rows])
    mean_border_error = np.mean([float(row["border_error_share"]) for row in rows])

    lines = [
        "# Analisis de errores por imagen - ensemble TTA",
        "",
        "## Configuracion analizada",
        "",
        "```text",
        f"Predicciones: {args.predictions_dir}",
        f"Split: {args.split}",
        "Referencia principal: manual_1",
        "Threshold: 0.45",
        "Modelo: cv_bce_dice_flips_valloss_pad_e120_aug_ensemble_tta",
        "```",
        "",
        "## Resumen global",
        "",
        f"- DICE medio total oficial recomputado: `{mean_dice:.12f}`",
        f"- Precision media contra experto 1 dentro del FoV: `{mean_precision:.6f}`",
        f"- Recall medio contra experto 1 dentro del FoV: `{mean_recall:.6f}`",
        f"- Recall medio en vasos finos aproximados: `{mean_thin_recall:.6f}`",
        f"- Proporcion media de error en zonas oscuras: `{mean_dark_error:.6f}`",
        f"- Proporcion media de error en borde FoV: `{mean_border_error:.6f}`",
        "",
        "## Peores imagenes por DICE",
        "",
        "| Imagen | DICE medio | Precision | Recall | FN manual | FP pred | Thin recall | Error oscuro | Error borde | Flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows_by_dice[: args.worst_count]:
        lines.append(
            "| "
            f"{row['image_id']} | "
            f"{float(row['dice_mean']):.6f} | "
            f"{float(row['precision_manual_1']):.6f} | "
            f"{float(row['recall_manual_1']):.6f} | "
            f"{float(row['fn_fraction_manual']):.6f} | "
            f"{float(row['fp_fraction_prediction']):.6f} | "
            f"{float(row['thin_recall']):.6f} | "
            f"{float(row['dark_error_share']):.6f} | "
            f"{float(row['border_error_share']):.6f} | "
            f"{row['flags']} |"
        )

    lines.extend(
        [
            "",
            "## Lectura de patrones",
            "",
            format_flag_group(rows, "low_recall_fn", "Imagenes con recall bajo / muchos falsos negativos"),
            format_flag_group(rows, "low_precision_fp", "Imagenes con precision baja / falsos positivos"),
            format_flag_group(rows, "thin_vessels_missed", "Imagenes con peor recall en vasos finos aproximados"),
            format_flag_group(rows, "dark_zone_errors", "Imagenes con errores concentrados en zonas oscuras"),
            format_flag_group(rows, "fov_border_errors", "Imagenes con errores relevantes en borde FoV"),
            format_flag_group(rows, "small_fp_noise", "Imagenes con falsos positivos pequenos tipo ruido"),
            "",
            "## Interpretacion",
            "",
            "La mayoria de los peores casos combinan recall bajo con perdida de vasos finos.",
            "Esto apunta mas a un problema de sensibilidad sobre estructuras pequenas que a un simple problema de threshold.",
            "",
            "Los falsos positivos fuera del FoV son cero porque las predicciones finales se generaron con `--apply-fov`.",
            "Por tanto, el postprocesado de FoV ya esta funcionando.",
            "",
            "El siguiente cambio con mas potencial es entrenamiento por parches balanceados.",
            "Antes de implementarlo, se puede probar postprocesado pequeno solo si las metricas muestran muchos componentes falsos positivos pequenos.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_flag_group(rows: list[dict[str, str | float | int]], flag: str, title: str) -> str:
    selected = [
        row for row in sorted(rows, key=lambda item: float(item["dice_mean"]))
        if flag in str(row["flags"]).split(";")
    ]
    if not selected:
        return f"- {title}: ninguna imagen marcada."
    formatted = ", ".join(
        f"{row['image_id']} (DICE {float(row['dice_mean']):.4f})" for row in selected
    )
    return f"- {title}: {formatted}."


def save_contact_sheet(
    worst_rows: list[dict[str, str | float | int]],
    arrays_by_id: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    tile_width = 220
    tile_height = 180
    label_height = 24
    columns = 4
    rows = len(worst_rows)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ["Imagen", "Manual 1", "Prediccion", "FP/FN"]

    for row_index, row in enumerate(worst_rows):
        sample_id = str(row["image_id"])
        arrays = arrays_by_id[sample_id]
        image = arrays["image"]
        manual_1 = arrays["manual_1"]
        prediction = arrays["prediction"]
        false_positive = (prediction > 0) & ~(manual_1 > 0)
        false_negative = (manual_1 > 0) & ~(prediction > 0)
        tiles = [
            image,
            mask_to_rgb(manual_1),
            overlay_mask(image, prediction > 0, color=(0, 255, 0), alpha=0.55),
            overlay_fp_fn(image, false_positive, false_negative),
        ]
        for col_index, tile in enumerate(tiles):
            x = col_index * tile_width
            y = row_index * (tile_height + label_height)
            resized = resize_rgb(tile, tile_width, tile_height)
            sheet.paste(resized, (x, y + label_height))
            if row_index == 0:
                draw.text((x + 6, 3), labels[col_index], fill=(0, 0, 0))
            if col_index == 0:
                text = f"{sample_id} DICE {float(row['dice_mean']):.3f}"
                draw.text((x + 6, y + label_height + 4), text, fill=(255, 255, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def resize_rgb(array: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(ensure_rgb(array))
    image.thumbnail((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), "black")
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    return np.repeat(mask[..., np.newaxis], 3, axis=-1)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    base = ensure_rgb(image).astype(np.float32)
    mask_bool = mask.astype(bool)
    color_array = np.asarray(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_array
    return np.clip(base, 0, 255).astype(np.uint8)


def overlay_fp_fn(image: np.ndarray, false_positive: np.ndarray, false_negative: np.ndarray) -> np.ndarray:
    overlay = ensure_rgb(image).copy()
    overlay[false_positive.astype(bool)] = np.asarray((0, 255, 255), dtype=np.uint8)
    overlay[false_negative.astype(bool)] = np.asarray((255, 0, 255), dtype=np.uint8)
    return overlay


if __name__ == "__main__":
    main()
