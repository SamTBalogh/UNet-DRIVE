from __future__ import annotations

import numpy as np
from PIL import Image


def add_flip_augmentations(
    images: np.ndarray,
    masks: np.ndarray,
    include_original: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synchronized horizontal/vertical flip augmentations.

    In segmentation, every geometric change applied to the image must also be
    applied to the mask, otherwise the target stops matching the input.
    """

    image_variants = []
    mask_variants = []

    if include_original:
        image_variants.append(images)
        mask_variants.append(masks)

    transforms = [
        (1,),  # vertical flip: height axis
        (2,),  # horizontal flip: width axis
        (1, 2),  # both flips
    ]

    for axes in transforms:
        image_variants.append(np.flip(images, axis=axes))
        mask_variants.append(np.flip(masks, axis=axes))

    return (
        np.concatenate(image_variants, axis=0).astype(np.float32),
        np.concatenate(mask_variants, axis=0).astype(np.float32),
    )


def add_rich_augmentations(
    images: np.ndarray,
    masks: np.ndarray,
    copies: int = 2,
    seed: int = 42,
    include_original: bool = True,
    rotation_degrees: float = 12.0,
    shift_pixels: int = 16,
    brightness_delta: float = 0.15,
    contrast_delta: float = 0.15,
    gamma_delta: float = 0.15,
    noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synchronized geometric and photometric augmentations.

    Geometric transforms are applied to image and mask. Photometric transforms
    are applied only to the image. Masks are always transformed with nearest
    neighbor interpolation and re-binarized to keep valid segmentation labels.
    """

    if len(images) != len(masks):
        raise ValueError("images and masks must contain the same number of samples")
    if copies < 0:
        raise ValueError("copies must be >= 0")

    rng = np.random.default_rng(seed)
    image_variants = []
    mask_variants = []

    if include_original:
        image_variants.append(images.astype(np.float32))
        mask_variants.append(masks.astype(np.float32))

    for _ in range(copies):
        augmented_images = []
        augmented_masks = []

        for image, mask in zip(images, masks):
            params = sample_transform_params(
                rng=rng,
                rotation_degrees=rotation_degrees,
                shift_pixels=shift_pixels,
                brightness_delta=brightness_delta,
                contrast_delta=contrast_delta,
                gamma_delta=gamma_delta,
                noise_std=noise_std,
            )
            aug_image = apply_geometric_transform(
                image,
                angle=params["angle"],
                shift_y=params["shift_y"],
                shift_x=params["shift_x"],
                is_mask=False,
            )
            aug_mask = apply_geometric_transform(
                mask,
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
            augmented_images.append(aug_image)
            augmented_masks.append(aug_mask)

        image_variants.append(np.stack(augmented_images).astype(np.float32))
        mask_variants.append(np.stack(augmented_masks).astype(np.float32))

    return (
        np.concatenate(image_variants, axis=0).astype(np.float32),
        np.concatenate(mask_variants, axis=0).astype(np.float32),
    )


def sample_transform_params(
    rng: np.random.Generator,
    rotation_degrees: float,
    shift_pixels: int,
    brightness_delta: float,
    contrast_delta: float,
    gamma_delta: float,
    noise_std: float,
) -> dict[str, float | int]:
    rotation = abs(float(rotation_degrees))
    shift = max(0, int(shift_pixels))
    brightness = abs(float(brightness_delta))
    contrast = abs(float(contrast_delta))
    gamma = abs(float(gamma_delta))
    noise = max(0.0, float(noise_std))

    return {
        "angle": float(rng.uniform(-rotation, rotation)) if rotation else 0.0,
        "shift_y": int(rng.integers(-shift, shift + 1)) if shift else 0,
        "shift_x": int(rng.integers(-shift, shift + 1)) if shift else 0,
        "brightness": float(rng.uniform(max(0.01, 1.0 - brightness), 1.0 + brightness)),
        "contrast": float(rng.uniform(max(0.01, 1.0 - contrast), 1.0 + contrast)),
        "gamma": float(rng.uniform(max(0.01, 1.0 - gamma), 1.0 + gamma)),
        "noise_std": float(rng.uniform(0.0, noise)) if noise else 0.0,
    }


def apply_geometric_transform(
    array: np.ndarray,
    angle: float,
    shift_y: int,
    shift_x: int,
    is_mask: bool,
) -> np.ndarray:
    pil_image = array_to_pil(array, is_mask=is_mask)
    fillcolor = 0
    if not is_mask and pil_image.mode == "RGB":
        fillcolor = (0, 0, 0)

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    transformed = pil_image.rotate(angle, resample=resample, fillcolor=fillcolor)
    transformed = transformed.transform(
        transformed.size,
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=resample,
        fillcolor=fillcolor,
    )
    channels = array.shape[-1] if array.ndim == 3 else 1
    return pil_to_array(transformed, is_mask=is_mask, channels=channels)


def apply_photometric_transform(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
    gamma: float,
    noise_std: float,
) -> np.ndarray:
    image = image.astype(np.float32)
    background_mask = np.all(image <= 1e-6, axis=-1) if image.ndim == 3 else image <= 1e-6

    transformed = image * brightness
    active = ~background_mask
    if np.any(active):
        mean = transformed[active].mean(axis=0)
    else:
        mean = transformed.mean(axis=(0, 1))
    transformed = (transformed - mean) * contrast + mean
    transformed = np.clip(transformed, 0.0, 1.0)
    transformed = np.power(transformed, gamma)

    if noise_std > 0:
        transformed = transformed + rng.normal(0.0, noise_std, size=transformed.shape)

    transformed = np.clip(transformed, 0.0, 1.0).astype(np.float32)
    transformed[background_mask] = 0.0
    return transformed


def array_to_pil(array: np.ndarray, is_mask: bool) -> Image.Image:
    array = np.asarray(array)
    if is_mask:
        if array.ndim == 3:
            array = array[..., 0]
        return Image.fromarray(((array > 0.5).astype(np.uint8)) * 255)

    image = np.clip(array, 0.0, 1.0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return Image.fromarray((image * 255.0).round().astype(np.uint8))


def pil_to_array(image: Image.Image, is_mask: bool, channels: int) -> np.ndarray:
    array = np.asarray(image)
    if is_mask:
        mask = (array > 127).astype(np.float32)
        return mask[..., np.newaxis]

    array = array.astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if channels == 1 and array.shape[-1] != 1:
        array = array[..., :1]
    return array.astype(np.float32)
