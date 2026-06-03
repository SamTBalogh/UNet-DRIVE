# DRIVE 2004 U-Net Vessel Segmentation

This project trains a convolutional neural network with a U-Net architecture to
segment blood vessels in retinal images from the DRIVE 2004 dataset.

The main technical goal is:

```text
input retinal image -> binary vessel mask
```

The project uses TensorFlow/Keras 3 and evaluates predictions with DICE score.

## Dataset

The expected local dataset path is:

```text
data/raw/DRIVE
```

Expected structure:

```text
data/raw/DRIVE/
  training/
    images/
    1st_manual/
    mask/
  test/
    images/
    1st_manual/
    2nd_manual/
    mask/
```

Important distinction:

- `1st_manual/` and `2nd_manual/` contain vessel masks.
- `mask/` contains the field-of-view mask, not the vessel ground truth.

The dataset is local-only and must not be committed to Git.

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- TensorFlow/Keras,
- NumPy,
- scikit-learn,
- Pillow,
- Matplotlib,
- tqdm.

## Initial Dataset Check

Before training, verify that each DRIVE image is correctly paired with its
manual vessel mask and field-of-view mask.

Training split:

```bash
python src/explore_drive.py --data-dir data/raw/DRIVE --split training
```

Test split:

```bash
python src/explore_drive.py --data-dir data/raw/DRIVE --split test
```

The script:

- prints the paired samples,
- saves a visual check figure under `outputs/figures/`.

Generated figures are local artifacts and are ignored by Git.

## Check DICE Implementation

Run:

```bash
python src/check_metrics.py
```

Expected output:

```text
same: 1.000000
different: 0.000000
partial: 0.500000
empty_vs_empty: 1.000000
```

This validates the DICE implementation with small artificial masks before using
it in training or evaluation.

## Pipeline Dry Run

Use a dry run to verify that data loading, preprocessing and model construction
work without starting training:

```bash
python src/train.py --dry-run --max-samples 2 --folds 1 --image-height 128 --image-width 128 --base-filters 4 --depth 2
```

The expected shapes are:

```text
X = (N, H, W, 3)
y = (N, H, W, 1)
```

Images should be normalized to `0-1`, and masks should contain only `0` and
`1`.

## Minimal Pilot Training

Run a very small pilot training to confirm that `model.fit()`, callbacks and
model saving work:

```bash
python src/train.py --max-samples 2 --folds 1 --image-height 64 --image-width 64 --base-filters 4 --depth 2 --epochs 1 --batch-size 1 --output-dir outputs/models/pilot_check
```

This is not intended to produce a good model. It only checks that the full
training path works.

Expected generated files:

```text
outputs/models/pilot_check/single_split.keras
outputs/models/pilot_check/single_split_history.csv
outputs/models/pilot_check/single_split_metadata.json
```

## Cross-Validation Training

Initial 5-fold training command:

```bash
python src/train.py --folds 5 --epochs 20 --batch-size 2 --augment-flips --output-dir outputs/models/cv_5folds
```

This trains one model per fold and saves them as:

```text
outputs/models/cv_5folds/fold_1.keras
outputs/models/cv_5folds/fold_2.keras
outputs/models/cv_5folds/fold_3.keras
outputs/models/cv_5folds/fold_4.keras
outputs/models/cv_5folds/fold_5.keras
```

## Generate PNG Segmentations

Example using one trained fold:

```bash
python src/predict.py --model outputs/models/cv_5folds/fold_1.keras --split test --output-dir outputs/segmentations/fold_1 --apply-fov
```

Predictions are saved as binary PNG masks:

```text
0   -> background
255 -> vessel
```

The `--apply-fov` flag forces pixels outside the DRIVE field of view to
background.

## Evaluate with DICE

Evaluate a trained model on the test split:

```bash
python src/evaluate.py --model outputs/models/cv_5folds/fold_1.keras --split test --output outputs/results/fold_1_test.csv --apply-fov
```

The CSV includes:

```text
image_id
model
threshold
dice_manual_1
dice_manual_2
dice_mean
```

For the DRIVE test split, both manual expert masks are used when available.

## Threshold Experiments

The default threshold is `0.5`. To compare thresholds:

```bash
python src/evaluate.py --model outputs/models/cv_5folds/fold_1.keras --split test --threshold 0.3 --output outputs/results/fold_1_test_t03.csv --apply-fov
python src/evaluate.py --model outputs/models/cv_5folds/fold_1.keras --split test --threshold 0.4 --output outputs/results/fold_1_test_t04.csv --apply-fov
python src/evaluate.py --model outputs/models/cv_5folds/fold_1.keras --split test --threshold 0.5 --output outputs/results/fold_1_test_t05.csv --apply-fov
python src/evaluate.py --model outputs/models/cv_5folds/fold_1.keras --split test --threshold 0.6 --output outputs/results/fold_1_test_t06.csv --apply-fov
```

For a rigorous final report, choose thresholds using validation data rather than
tuning directly on the test set.

## Project Structure

```text
src/
  data.py              # DRIVE loading and preprocessing
  explore_drive.py     # dataset pairing and visual checks
  metrics.py           # DICE metrics and Dice loss
  check_metrics.py     # artificial-mask metric test
  model.py             # configurable U-Net
  augment.py           # synchronized flip augmentation
  train.py             # pilot training and K-fold training
  predict.py           # PNG segmentation generation
  evaluate.py          # DICE evaluation and CSV output
  config.py            # shared default settings
```

## Local Artifacts

The following paths are local-only and ignored by Git:

```text
data/raw/
data/processed/
outputs/
docs/
report/
*.keras
*.h5
*.ckpt
*.npy
*.npz
```

The final delivery ZIP may include trained models and PNG segmentations, but
during development they should remain outside Git.

## TensorFlow on Windows

TensorFlow may print this warning on native Windows:

```text
TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.
```

This means TensorFlow is likely training on CPU. For long 5-fold training runs,
WSL2 is recommended if an NVIDIA GPU is available.

