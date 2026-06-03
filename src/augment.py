from __future__ import annotations

import numpy as np


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
