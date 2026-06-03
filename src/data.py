from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VALID_SPLITS = {"training", "test"}
DEFAULT_IMAGE_SIZE = (512, 512)


@dataclass(frozen=True)
class DriveSample:
    """Paths associated with one DRIVE image."""

    sample_id: str
    split: str
    image_path: Path
    manual_1_path: Path
    fov_mask_path: Path
    manual_2_path: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "id": self.sample_id,
            "split": self.split,
            "image": str(self.image_path),
            "manual_1": str(self.manual_1_path),
            "manual_2": str(self.manual_2_path) if self.manual_2_path else None,
            "fov_mask": str(self.fov_mask_path),
        }


def list_drive_samples(
    data_dir: str | Path,
    split: str = "training",
    require_manual_2: bool | None = None,
) -> list[DriveSample]:
    """Return DRIVE image/manual/FoV mask triplets for a split.

    Expected structure:

    data_dir/
      training/images/21_training.tif
      training/1st_manual/21_manual1.gif
      training/mask/21_training_mask.gif
      test/images/01_test.tif
      test/1st_manual/01_manual1.gif
      test/2nd_manual/01_manual2.gif
      test/mask/01_test_mask.gif
    """

    if split not in VALID_SPLITS:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Invalid split '{split}'. Expected one of: {valid}")

    root = Path(data_dir)
    split_dir = root / split
    images_dir = split_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"DRIVE images folder not found: {images_dir}")

    if require_manual_2 is None:
        require_manual_2 = split == "test"

    image_suffix = "training" if split == "training" else "test"
    image_paths = sorted(images_dir.glob(f"*_{image_suffix}.tif"), key=_path_sample_number)

    samples: list[DriveSample] = []
    missing_files: list[str] = []

    for image_path in image_paths:
        sample_id = _sample_id_from_path(image_path)
        manual_1_path = split_dir / "1st_manual" / f"{sample_id}_manual1.gif"
        manual_2_path = split_dir / "2nd_manual" / f"{sample_id}_manual2.gif"
        fov_mask_path = split_dir / "mask" / f"{sample_id}_{image_suffix}_mask.gif"

        for required_path in (manual_1_path, fov_mask_path):
            if not required_path.exists():
                missing_files.append(str(required_path))

        has_manual_2 = manual_2_path.exists()
        if require_manual_2 and not has_manual_2:
            missing_files.append(str(manual_2_path))

        samples.append(
            DriveSample(
                sample_id=sample_id,
                split=split,
                image_path=image_path,
                manual_1_path=manual_1_path,
                manual_2_path=manual_2_path if has_manual_2 else None,
                fov_mask_path=fov_mask_path,
            )
        )

    if missing_files:
        formatted = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing DRIVE files:\n{formatted}")

    return samples


def load_drive_sample(sample: DriveSample) -> dict[str, np.ndarray | None]:
    """Load a DRIVE sample as arrays without changing value ranges."""

    return {
        "image": read_image(sample.image_path),
        "manual_1": read_image(sample.manual_1_path),
        "manual_2": read_image(sample.manual_2_path) if sample.manual_2_path else None,
        "fov_mask": read_image(sample.fov_mask_path),
    }


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file with Pillow and return it as a NumPy array."""

    with Image.open(path) as image:
        return np.asarray(image.copy())


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert an image from 0-255 uint values to float32 values in 0-1."""

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return image


def binarize_mask(mask: np.ndarray, threshold: int | float = 127) -> np.ndarray:
    """Convert a grayscale mask to a binary uint8 array with values 0 and 1."""

    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > threshold).astype(np.uint8)


def ensure_channel_last(array: np.ndarray) -> np.ndarray:
    """Ensure an array has an explicit final channel dimension."""

    if array.ndim == 2:
        return array[..., np.newaxis]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Prepare an image for Keras: channel-last float32 in range 0-1."""

    image = ensure_channel_last(image)
    return normalize_image(image).astype(np.float32)


def prepare_mask(mask: np.ndarray, threshold: int | float = 127) -> np.ndarray:
    """Prepare a binary segmentation mask with shape (H, W, 1)."""

    mask = binarize_mask(mask, threshold=threshold)
    return ensure_channel_last(mask).astype(np.float32)


def resize_array(
    array: np.ndarray,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    is_mask: bool = False,
) -> np.ndarray:
    """Resize an image or mask to (height, width).

    Masks use nearest-neighbor interpolation so their labels remain binary.
    """

    height, width = size
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR

    if array.dtype == bool:
        array = array.astype(np.uint8)

    with Image.fromarray(array) as image:
        resized = image.resize((width, height), resample=resampling)
        return np.asarray(resized)


def load_preprocessed_sample(
    sample: DriveSample,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, np.ndarray | None]:
    """Load one DRIVE sample already prepared for model training/evaluation."""

    arrays = load_drive_sample(sample)
    image = resize_array(arrays["image"], size=image_size, is_mask=False)
    manual_1 = resize_array(arrays["manual_1"], size=image_size, is_mask=True)
    fov_mask = resize_array(arrays["fov_mask"], size=image_size, is_mask=True)

    manual_2 = None
    if arrays["manual_2"] is not None:
        manual_2 = resize_array(arrays["manual_2"], size=image_size, is_mask=True)

    return {
        "image": prepare_image(image),
        "manual_1": prepare_mask(manual_1),
        "manual_2": prepare_mask(manual_2) if manual_2 is not None else None,
        "fov_mask": prepare_mask(fov_mask),
    }


def load_drive_arrays(
    samples: list[DriveSample],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    target: str = "manual_1",
) -> tuple[np.ndarray, np.ndarray]:
    """Load multiple DRIVE samples into X and y arrays.

    This simple in-memory loader is enough for DRIVE because the dataset is
    small. For larger datasets, a streaming `tf.data.Dataset` would be better.
    """

    if target not in {"manual_1", "manual_2"}:
        raise ValueError("target must be 'manual_1' or 'manual_2'")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for sample in samples:
        arrays = load_preprocessed_sample(sample, image_size=image_size)
        mask = arrays[target]
        if mask is None:
            raise ValueError(f"Sample {sample.sample_id} does not have {target}")
        images.append(arrays["image"])
        masks.append(mask)

    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.float32)


def _sample_id_from_path(path: Path) -> str:
    match = re.match(r"(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot extract DRIVE sample id from filename: {path.name}")
    return match.group(1)


def _path_sample_number(path: Path) -> int:
    return int(_sample_id_from_path(path))
