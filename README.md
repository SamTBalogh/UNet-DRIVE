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

Next recommended experiment after the `cv_5folds` baseline:

```bash
TF_GPU_ALLOCATOR=cuda_malloc_async python src/train.py --folds 5 --epochs 60 --batch-size 1 --augment-flips --loss bce_dice --output-dir outputs/models/cv_bce_dice
```

`--loss bce_dice` combines binary cross-entropy with Dice loss, which is more
aligned with the final DICE objective on imbalanced vessel masks. `batch-size 1`
is the safer default for WSL2/GPU runs on limited VRAM; if memory is stable, try
`--batch-size 2`.

To avoid deforming DRIVE images into a square input, train with symmetric
padding instead of resizing:

```bash
TF_GPU_ALLOCATOR=cuda_malloc_async TF_FORCE_GPU_ALLOW_GROWTH=true python src/train.py --folds 5 --epochs 80 --batch-size 1 --augment-flips --loss bce_dice --checkpoint-monitor val_loss --checkpoint-mode min --early-stopping-monitor val_loss --early-stopping-mode min --patience 12 --resize-strategy pad --output-dir outputs/models/cv_bce_dice_flips_valloss_pad
```

For DRIVE, `--resize-strategy pad` resolves the model input size to `592x576`
with the default `--pad-multiple 16`. Metadata is saved next to each fold so
evaluation, threshold tuning and ensemble scripts can infer the preprocessing.

After the padded 120-epoch ensemble, the next controlled improvement is richer
augmentation while keeping the rest of the experiment fixed:

```bash
TF_GPU_ALLOCATOR=cuda_malloc_async TF_FORCE_GPU_ALLOW_GROWTH=true python src/train.py --folds 5 --epochs 120 --batch-size 1 --augment-flips --augment-rich --augment-rich-copies 2 --loss bce_dice --checkpoint-monitor val_loss --checkpoint-mode min --early-stopping-monitor val_loss --early-stopping-mode min --patience 16 --resize-strategy pad --output-dir outputs/models/cv_bce_dice_flips_valloss_pad_e120_aug
```

`--augment-rich` adds synchronized small rotations and translations to image and
mask, plus image-only brightness, contrast, gamma and light noise changes. Masks
are transformed with nearest-neighbor interpolation and re-binarized.

## Balanced Patch Training

The next higher-potential experiment after error analysis is balanced patch
training. It trains the same U-Net on sampled `128x128` patches instead of full
images, with batches biased toward vessel, thin-vessel and low-vessel patches.
The split is still done by original image before patch extraction, so validation
patches never come from training images.

Recommended first full run:

```bash
TF_GPU_ALLOCATOR=cuda_malloc_async TF_FORCE_GPU_ALLOW_GROWTH=true python src/train_patches.py --folds 5 --patch-size 128 --candidate-stride 16 --patches-per-image 512 --val-patches-per-image 256 --batch-size 8 --epochs 120 --augment-flips --augment-rich --loss bce_dice --checkpoint-monitor val_loss --checkpoint-mode min --early-stopping-monitor val_loss --early-stopping-mode min --patience 16 --output-dir outputs/models/cv_bce_dice_patch128_balanced
```

Run a fast dry run before training on a new machine:

```bash
python src/train_patches.py --dry-run --max-samples 2 --folds 1 --patch-size 64 --candidate-stride 32 --patches-per-image 8 --val-patches-per-image 4 --batch-size 2 --base-filters 4 --depth 2 --augment-flips --augment-rich --output-dir outputs/models/patch_dry_run
```

Patch-trained models require sliding-window inference for full-image evaluation.
Do not evaluate them with `evaluate_ensemble.py`; use the patch ensemble scripts
instead:

```bash
python src/tune_threshold_patch_cv.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --output-dir outputs/results/cv_bce_dice_patch128_balanced --apply-fov --predict-batch-size 32
python src/evaluate_patch_ensemble.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --split test --threshold 0.45 --output outputs/results/cv_bce_dice_patch128_balanced/patch_ensemble_test_t045.csv --ensemble-name cv_bce_dice_patch128_balanced_ensemble --apply-fov --predict-batch-size 32
python src/evaluate_patch_ensemble.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --split test --threshold 0.45 --output outputs/results/cv_bce_dice_patch128_balanced/patch_ensemble_test_t045_tta.csv --ensemble-name cv_bce_dice_patch128_balanced_ensemble_tta --apply-fov --tta --predict-batch-size 32
python src/evaluate_patch_ensemble.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --split test --threshold 0.45 --output outputs/results/cv_bce_dice_patch128_balanced_stride32_tta/patch_ensemble_test_t045_stride32_tta.csv --ensemble-name cv_bce_dice_patch128_balanced_ensemble_stride32_tta --apply-fov --tta --stride 32 --predict-batch-size 32
```

Current best observed result:

```text
cv_bce_dice_patch128_balanced_ensemble_stride32_tta
threshold = 0.45
patch_size = 128
stride = 32
TTA = enabled
test mean DICE total = 0.832967677712
```

The stride 32 result is the maximum-quality configuration. If inference speed is
more important, `cv_bce_dice_patch128_balanced_ensemble_tta` with stride 64 is
nearly equivalent (`0.832503151894` test mean DICE total).

Next controlled experiments should be run one at a time:

- conservative connected-component postprocessing inside the FoV, validated on
  CV before looking at test;
- a thin-vessel-oriented loss such as weighted BCE + Dice, Focal/Tversky, or a
  Dice variant with a thin-vessel proxy weight.

## Check Saved Model Loading

Before delivering or evaluating final models, verify that all saved `.keras`
files can be loaded with Keras 3:

```bash
python src/check_model_load.py --models-dir outputs/models/cv_5folds
```

This check defaults to CPU so it does not consume GPU memory in WSL2. If you
explicitly want to inspect GPU visibility, use:

```bash
python src/check_model_load.py --models-dir outputs/models/cv_5folds --device gpu --cuda-malloc-async
```

The current models were compiled with the custom `dice_coef` metric. They load
after importing `metrics.py`, and they also load with `compile=False`. A clean
plain `load_model(path)` without importing project metrics is expected to fail
for these compiled models.

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

To generate segmentations with an ensemble of the five cross-validation folds:

```bash
python src/predict_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss --split test --threshold 0.50 --output-dir outputs/segmentations/final_ensemble_test --apply-fov
```

To create visual diagnostics for selected test images:

```bash
python src/diagnose_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss --split test --threshold 0.50 --output-dir outputs/figures/final_ensemble_diagnostics --apply-fov
```

For patch-trained ensembles, use sliding-window generation:

```bash
python src/predict_patch_ensemble.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --split test --threshold 0.45 --output-dir outputs/segmentations/patch_ensemble_test_t045_stride32_tta --apply-fov --tta --stride 32 --predict-batch-size 32
python src/diagnose_patch_ensemble.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --split test --threshold 0.45 --output-dir outputs/figures/patch_ensemble_diagnostics_t045_stride32_tta --sample-ids 01,03,07,14,20 --apply-fov --tta --stride 32 --predict-batch-size 32
```

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

Evaluate an ensemble by averaging fold probabilities before thresholding:

```bash
python src/evaluate_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss --split test --threshold 0.50 --output outputs/results/final_ensemble_test.csv --apply-fov
```

The ensemble script also writes a summary CSV next to the per-image results.

To test reversible TTA on the current best ensemble, add `--tta`. This predicts
the original image, horizontal flip, vertical flip and both flips, reverses each
flipped probability map, averages the four probabilities per fold, and then
averages the folds before thresholding:

```bash
python src/evaluate_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss_pad_e120_aug --split test --threshold 0.45 --output outputs/results/cv_bce_dice_flips_valloss_pad_e120_aug/ensemble_test_t045_tta.csv --ensemble-name cv_bce_dice_flips_valloss_pad_e120_aug_ensemble_tta --apply-fov --tta
```

Use the same flag for TTA segmentations or diagnostics:

```bash
python src/predict_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss_pad_e120_aug --split test --threshold 0.45 --output-dir outputs/segmentations/final_ensemble_test_pad_e120_aug_tta --apply-fov --tta
python src/diagnose_ensemble.py --models-dir outputs/models/cv_bce_dice_flips_valloss_pad_e120_aug --split test --threshold 0.45 --output-dir outputs/figures/final_ensemble_diagnostics_pad_e120_aug_tta --apply-fov --tta
```

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

For the saved 5-fold models, tune thresholds on each fold's validation split
using the stored metadata:

```bash
python src/tune_threshold_cv.py --models-dir outputs/models/cv_5folds --output-dir outputs/results/cv_5folds --apply-fov
```

Add `--tta` to repeat the same validation-threshold search with reversible TTA.

For padded models, use the same command shape; preprocessing is read from fold
metadata:

```bash
python src/tune_threshold_cv.py --models-dir outputs/models/cv_bce_dice_flips_valloss_pad --output-dir outputs/results/cv_bce_dice_flips_valloss_pad --apply-fov
```

For patch-trained models, tune thresholds with full-image sliding-window
validation:

```bash
python src/tune_threshold_patch_cv.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --output-dir outputs/results/cv_bce_dice_patch128_balanced --apply-fov --predict-batch-size 32
python src/tune_threshold_patch_cv.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --output-dir outputs/results/cv_bce_dice_patch128_balanced_tta --apply-fov --tta --predict-batch-size 32
python src/tune_threshold_patch_cv.py --models-dir outputs/models/cv_bce_dice_patch128_balanced --output-dir outputs/results/cv_bce_dice_patch128_balanced_stride32_tta --apply-fov --tta --stride 32 --predict-batch-size 32
```

## Project Structure

```text
src/
  data.py              # DRIVE loading and preprocessing
  explore_drive.py     # dataset pairing and visual checks
  metrics.py           # DICE metrics and Dice loss
  check_metrics.py     # artificial-mask metric test
  model.py             # configurable U-Net
  augment.py           # synchronized flip and richer augmentation
  train.py             # pilot training and K-fold training
  check_model_load.py  # Keras 3 saved-model loading check
  tune_threshold_cv.py # threshold search on CV validation folds
  ensemble.py          # shared ensemble inference helpers
  predict.py           # PNG segmentation generation
  predict_ensemble.py  # PNG generation with averaged fold probabilities
  diagnose_ensemble.py # visual diagnostics for ensemble predictions
  evaluate.py          # DICE evaluation and CSV output
  evaluate_ensemble.py # DICE evaluation with averaged fold probabilities
  patches.py           # patch candidate extraction and balanced sampling
  train_patches.py     # balanced patch training
  patch_inference.py   # sliding-window patch inference helpers
  tune_threshold_patch_cv.py # threshold search for patch-trained CV folds
  evaluate_patch_ensemble.py # DICE evaluation for patch-trained ensembles
  predict_patch_ensemble.py  # PNG generation for patch-trained ensembles
  diagnose_patch_ensemble.py # visual diagnostics for patch-trained ensembles
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
