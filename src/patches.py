from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data import DriveSample, binarize_mask, load_drive_sample, normalize_image


PATCH_CATEGORIES = ("positive", "thin", "background")


@dataclass(frozen=True)
class PatchSamplingConfig:
    patch_size: int = 128
    candidate_stride: int = 16
    min_fov_ratio: float = 0.50
    positive_ratio_min: float = 0.01
    background_ratio_max: float = 0.002
    thin_ratio_min: float = 0.001
    thin_neighbor_threshold: int = 4
    positive_fraction: float = 0.60
    thin_fraction: float = 0.20
    background_fraction: float = 0.20

    def category_fractions(self) -> dict[str, float]:
        return {
            "positive": self.positive_fraction,
            "thin": self.thin_fraction,
            "background": self.background_fraction,
        }


@dataclass
class PatchSource:
    sample_id: str
    image: np.ndarray
    mask: np.ndarray
    fov: np.ndarray
    thin_mask: np.ndarray
    patch_size: int
    candidates: dict[str, list[tuple[int, int]]]


def build_patch_sources(
    samples: list[DriveSample],
    config: PatchSamplingConfig,
) -> list[PatchSource]:
    """Load DRIVE samples and precompute patch candidates by category."""

    sources = []
    for sample in samples:
        arrays = load_drive_sample(sample)
        image = prepare_image(arrays["image"])
        mask = binarize_mask(arrays["manual_1"]).astype(np.uint8)
        fov = binarize_mask(arrays["fov_mask"]).astype(np.uint8)

        image, mask, fov = pad_to_patch_size(image, mask, fov, patch_size=config.patch_size)
        thin_mask = thin_vessel_proxy(mask, threshold=config.thin_neighbor_threshold)
        candidates = categorize_patch_candidates(
            mask=mask,
            fov=fov,
            thin_mask=thin_mask,
            config=config,
        )
        sources.append(
            PatchSource(
                sample_id=sample.sample_id,
                image=image,
                mask=mask,
                fov=fov,
                thin_mask=thin_mask,
                patch_size=config.patch_size,
                candidates=candidates,
            )
        )
    return sources


def prepare_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return normalize_image(image).astype(np.float32)


def pad_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    if pad_height == 0 and pad_width == 0:
        return image, mask, fov

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    image_pad = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    mask_pad = [(pad_top, pad_bottom), (pad_left, pad_right)]
    return (
        np.pad(image, image_pad, mode="constant", constant_values=0).astype(np.float32),
        np.pad(mask, mask_pad, mode="constant", constant_values=0).astype(np.uint8),
        np.pad(fov, mask_pad, mode="constant", constant_values=0).astype(np.uint8),
    )


def categorize_patch_candidates(
    mask: np.ndarray,
    fov: np.ndarray,
    thin_mask: np.ndarray,
    config: PatchSamplingConfig,
) -> dict[str, list[tuple[int, int]]]:
    patch_size = config.patch_size
    candidates: dict[str, list[tuple[int, int]]] = {
        "positive": [],
        "thin": [],
        "background": [],
        "all": [],
    }
    scored_candidates: list[tuple[tuple[int, int], float, float]] = []

    for top in candidate_axis_positions(mask.shape[0], patch_size, config.candidate_stride):
        for left in candidate_axis_positions(mask.shape[1], patch_size, config.candidate_stride):
            fov_patch = fov[top : top + patch_size, left : left + patch_size] > 0
            fov_pixels = int(np.sum(fov_patch))
            fov_ratio = fov_pixels / float(patch_size * patch_size)
            if fov_ratio < config.min_fov_ratio or fov_pixels == 0:
                continue

            mask_patch = mask[top : top + patch_size, left : left + patch_size] > 0
            thin_patch = thin_mask[top : top + patch_size, left : left + patch_size] > 0
            vessel_ratio = float(np.sum(mask_patch & fov_patch)) / float(fov_pixels)
            thin_ratio = float(np.sum(thin_patch & fov_patch)) / float(fov_pixels)
            location = (top, left)
            candidates["all"].append(location)
            scored_candidates.append((location, vessel_ratio, thin_ratio))

            if vessel_ratio >= config.positive_ratio_min:
                candidates["positive"].append(location)
            if (
                thin_ratio >= config.thin_ratio_min
                or config.background_ratio_max < vessel_ratio < config.positive_ratio_min
            ):
                candidates["thin"].append(location)
            if vessel_ratio <= config.background_ratio_max:
                candidates["background"].append(location)

    if not candidates["all"]:
        raise ValueError("No valid patch candidates found. Check FoV and patch settings.")

    fill_empty_candidate_categories(candidates, scored_candidates)
    return candidates


def fill_empty_candidate_categories(
    candidates: dict[str, list[tuple[int, int]]],
    scored_candidates: list[tuple[tuple[int, int], float, float]],
) -> None:
    fallback_count = max(1, int(np.ceil(len(scored_candidates) * 0.25)))
    if not candidates["positive"]:
        by_vessel_desc = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        candidates["positive"] = [location for location, _vessel, _thin in by_vessel_desc[:fallback_count]]
    if not candidates["thin"]:
        by_thin_desc = sorted(scored_candidates, key=lambda item: (item[2], -item[1]), reverse=True)
        candidates["thin"] = [location for location, _vessel, _thin in by_thin_desc[:fallback_count]]
    if not candidates["background"]:
        by_vessel_asc = sorted(scored_candidates, key=lambda item: item[1])
        candidates["background"] = [location for location, _vessel, _thin in by_vessel_asc[:fallback_count]]


def candidate_axis_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("candidate_stride must be positive")
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def thin_vessel_proxy(mask: np.ndarray, threshold: int) -> np.ndarray:
    counts = neighbor_count(mask > 0)
    return ((mask > 0) & (counts <= threshold)).astype(np.uint8)


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    counts = np.zeros_like(mask, dtype=np.uint8)
    for row_offset in range(3):
        for col_offset in range(3):
            counts += padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return counts


def sample_balanced_patch_batch(
    sources: list[PatchSource],
    batch_size: int,
    rng: np.random.Generator,
    category_fractions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str | int]]]:
    if not sources:
        raise ValueError("At least one PatchSource is required.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    category_counts = allocate_category_counts(
        batch_size=batch_size,
        category_fractions=category_fractions,
    )
    images = []
    masks = []
    metadata = []
    for category, count in category_counts.items():
        for _ in range(count):
            source = choose_source_with_category(sources, category=category, rng=rng)
            top, left = choose_candidate(source, category=category, rng=rng)
            images.append(crop_image_patch(source.image, top, left, patch_size=source.patch_size))
            masks.append(crop_mask_patch(source.mask, top, left, patch_size=source.patch_size))
            metadata.append(
                {
                    "sample_id": source.sample_id,
                    "category": category,
                    "top": top,
                    "left": left,
                }
            )

    order = rng.permutation(len(images))
    x = np.stack([images[index] for index in order]).astype(np.float32)
    y = np.stack([masks[index] for index in order]).astype(np.float32)
    metadata = [metadata[index] for index in order]
    return x, y, metadata


def sample_balanced_patch_arrays(
    sources: list[PatchSource],
    total_patches: int,
    batch_size: int,
    rng: np.random.Generator,
    category_fractions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    masks = []
    remaining = total_patches
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        x_batch, y_batch, _metadata = sample_balanced_patch_batch(
            sources=sources,
            batch_size=current_batch,
            rng=rng,
            category_fractions=category_fractions,
        )
        images.append(x_batch)
        masks.append(y_batch)
        remaining -= current_batch
    return np.concatenate(images, axis=0), np.concatenate(masks, axis=0)


def allocate_category_counts(
    batch_size: int,
    category_fractions: dict[str, float],
) -> dict[str, int]:
    fractions = {category: max(0.0, float(category_fractions.get(category, 0.0))) for category in PATCH_CATEGORIES}
    total = sum(fractions.values())
    if total <= 0:
        fractions = {category: 1.0 / len(PATCH_CATEGORIES) for category in PATCH_CATEGORIES}
    else:
        fractions = {category: value / total for category, value in fractions.items()}

    raw_counts = {category: fractions[category] * batch_size for category in PATCH_CATEGORIES}
    counts = {category: int(np.floor(raw_counts[category])) for category in PATCH_CATEGORIES}
    remainder = batch_size - sum(counts.values())
    by_remainder = sorted(
        PATCH_CATEGORIES,
        key=lambda category: raw_counts[category] - counts[category],
        reverse=True,
    )
    for category in by_remainder[:remainder]:
        counts[category] += 1
    return counts


def choose_source_with_category(
    sources: list[PatchSource],
    category: str,
    rng: np.random.Generator,
) -> PatchSource:
    weights = np.asarray([len(source.candidates.get(category, [])) for source in sources], dtype=np.float64)
    if np.sum(weights) <= 0:
        weights = np.asarray([len(source.candidates["all"]) for source in sources], dtype=np.float64)
    probabilities = weights / np.sum(weights)
    return sources[int(rng.choice(len(sources), p=probabilities))]


def choose_candidate(
    source: PatchSource,
    category: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    candidates = source.candidates.get(category) or source.candidates["all"]
    return candidates[int(rng.integers(0, len(candidates)))]


def crop_image_patch(
    image: np.ndarray,
    top: int,
    left: int,
    patch_size: int,
) -> np.ndarray:
    return image[top : top + patch_size, left : left + patch_size, :].astype(np.float32)


def crop_mask_patch(mask: np.ndarray, top: int, left: int, patch_size: int) -> np.ndarray:
    patch = mask[top : top + patch_size, left : left + patch_size].astype(np.float32)
    return patch[..., np.newaxis]


def summarize_patch_sources(sources: list[PatchSource]) -> list[dict[str, str | int]]:
    rows = []
    for source in sources:
        rows.append(
            {
                "sample_id": source.sample_id,
                "positive_candidates": len(source.candidates["positive"]),
                "thin_candidates": len(source.candidates["thin"]),
                "background_candidates": len(source.candidates["background"]),
                "all_candidates": len(source.candidates["all"]),
            }
        )
    return rows


def write_patch_source_summary(
    sources: list[PatchSource],
    output_path: str | Path,
) -> None:
    import csv

    rows = summarize_patch_sources(sources)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_id",
            "positive_candidates",
            "thin_candidates",
            "background_candidates",
            "all_candidates",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
