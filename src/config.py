from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("data/raw/DRIVE")
OUTPUTS_DIR = Path("outputs")

IMAGE_SIZE = (512, 512)
RANDOM_SEED = 42

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_FOLDS = 5
