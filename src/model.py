from __future__ import annotations

from keras import Model, layers


def conv_block(x, filters: int, dropout: float = 0.0):
    """Two convolution layers used repeatedly by U-Net."""

    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (512, 512, 3),
    base_filters: int = 16,
    depth: int = 4,
    dropout: float = 0.0,
) -> Model:
    """Build a compact U-Net for binary retinal vessel segmentation."""

    if depth < 2:
        raise ValueError("depth must be at least 2")

    divisor = 2**depth
    height, width, _channels = input_shape
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"input height and width must be divisible by {divisor} for depth={depth}"
        )

    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    for level in range(depth):
        filters = base_filters * (2**level)
        x = conv_block(x, filters=filters, dropout=dropout)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_block(x, filters=base_filters * (2**depth), dropout=dropout)

    for level in reversed(range(depth)):
        filters = base_filters * (2**level)
        x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block(x, filters=filters, dropout=dropout)

    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs, name="drive_unet")
