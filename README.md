# Detection and Prediction of Freezing of Gait

This repository contains all the notebooks for the project, split into two parts:

1. **`FoG-STAR/`** — model development and refinement on the FoG-STAR dataset (detection + pre-FoG prediction).
2. **`Brazil/`** — labeling pipeline for the new dataset (the "Brazil dataset"), testing the FoG-STAR-trained model on it and retraining/validating a model directly on it with Leave-One-Subject-Out (LOSO) cross-validation.

---

# Part 1 - FoG-STAR


## Overview

The pipeline evolves through three stages:

1. **Data exploration** — understanding the FoG-STAR dataset and the difficulty of the train/val/test split.
2. **Old CNN (TensorFlow)** — reproducing and testing a baseline single/dual-sensor 1D-CNN from the literature.
3. **New CNN (PyTorch, multi-branch)** — a custom multi-branch architecture (ankle + back sensors) developed to improve on the baseline, followed by class-imbalance handling, cross-validation, alternative input features, data augmentation, post-processing and finally extension to **pre-FoG prediction**.

### Data files referenced across notebooks
- `sensor_data.csv` — raw FoG-STAR sensor data (used in `FoG_Star_Analytics`)
- `sensor_data_quaternions.csv` — sensor data with quaternion-transformed orientation channels (used in most "New CNN" notebooks)
- `sensor_data_complete.csv` — extended sensor data version (used in `New_model_crossval` and `New_model_accel`)

### Train/val/test splits
Two subject-level splits were used over the course of the project (see `activity_distribution.ipynb`):
- **Old split**: train = subjects 1–11, val = 12–16, test = 17–22
- **New split**: train = `[2,3,10,12,13,14,16,17,18,21]`, val = `[1,5,6,7,11,15,20]`, test = `[4,8,9,19,22]`

The new split was adopted after the analysis in `activity_distribution.ipynb` showed the old split unevenly distributed FoG events (and difficult activities) across sets. Most "New CNN" notebooks use the new split; check the data-loading cell at the top of each notebook to confirm which one it uses.

---

## Notebooks

### 1. Data exploration

**`FoG_Star_Analytics.ipynb`**
Exploratory analysis of `sensor_data.csv`. Generates the dataset's descriptive figures: activity distribution, FoG vs. non-FoG sample counts, FoG event duration distributions and FoG event durations broken down by severity (Shuffling / Trembling / Akinesia).

**`activity_distribution.ipynb`**
Compares how activities and FoG samples are distributed across the train/validation/test sets, for both the old and new subject splits. Includes an analysis (Section 7) of which subjects contribute the most FoG samples in the most difficult activities (activities 1 and 3), which motivated the move to the new split.

### 2. Old CNN (TensorFlow baseline)

**`test_baseline.ipynb`**
Tests the baseline 1D-CNN model (TensorFlow/Keras) from the reference article, trained on gyroscope data. Evaluates the model on: right ankle alone, left ankle alone and a two-leg fusion (max-rule). Includes a detection-latency analysis, an attempt at activity-gating using the back sensor (which hurt recall and was abandoned), per-subject fine-tuning (subject 19) and a final cross-analysis of FoG events by activity.

**`test_quaternions_baseline.ipynb`**
Rebuilds the input features as quaternions (12 channels: ankle L/R + back, q0–q3) and retests the baseline-style model. First tests **absolute quaternions** — found to generalize poorly to unseen subjects/orientations due to the model overfitting to absolute heading/orientation. Then tests **relative quaternions** (orientation relative to the start of each window), which fixes this generalization failure.

**`CNN_ankle_back.ipynb`**
Extends the baseline 1D-CNN to 9 input channels (ankle L/R + lumbar/back gyroscopes) to directly test whether adding back/trunk context improves on the single/dual-ankle-sensor setup. Includes a per-error analysis of the fused 9-channel model.

### 3. New CNN (PyTorch multi-branch)

**`specific_analysis.ipynb`**
Introduces the **`MultiBranchCNN`** architecture used for the rest of the project: a separate 1D-CNN branch for the ankle sensors (8 channels: L+R quaternions) and a separate branch for the back sensor (4 channels: quaternions), fused before the final classifier. Covers training, evaluation, PCA/t-SNE visualization of learned features, per-subject analysis, class-imbalance analysis and performance broken down by FoG severity level. Uses the old split.

**`New_model_crossval.ipynb`**
Same `MultiBranchCNN` architecture, retrained on the new split using `sensor_data_complete.csv`. Adds **Grouped K-Fold cross-validation** (grouped by subject) to get a more robust estimate of performance and a final held-out test evaluation.

**`New_model_sensors.ipynb`**
Adapts the `MultiBranchCNN` to a different input channel composition (ankle branch: 6 channels, back branch: 3 channels — gyroscope per axis only, rather than the 8/4-channel quaternion version. Adds a `FocalLoss` option for class imbalance.

### 4. Robustness and refinement

**`data_augmentation.ipynb`**
Tests data augmentation strategies on the `MultiBranchCNN`/new-split setup: global augmentation, FoG-only augmentation and FoG + hard-activity augmentation (activities 1 and 3 specifically). Compares "best validation checkpoint" vs. "last epoch checkpoint." Also includes per-subject and per-activity performance breakdowns and a test of **personalized per-activity decision thresholds** instead of a single global threshold.

**`test_post_processing.ipynb`**
Trains the model (1-second windows/2-seconds windows) and tests several post-processing strategies applied to the raw output probabilities/predictions: per-activity thresholding, **probability smoothing**, **temporal smoothing** (moving average on binary predictions), and **hysteresis thresholding** (separate low/high thresholds to reduce rapid 0/1 oscillation). Ends with a systematic comparison of combinations of these methods against the no-post-processing baseline.

### 5. Pre-FoG prediction

**`pre_fog_exploration.ipynb`**
Exploratory step asking whether the *trained detection model's* output probability rises in the seconds **before** a FoG event actually starts. Identifies FoG onsets (0→1 transitions), extracts the probability trajectory in a lookback window before each onset, visualizes and quantifies the trend, and tests a simple heuristic ("if probability > threshold at t-1, predict FoG at t") as an early baseline for prediction.

**`pre_fog_model.ipynb`**
Builds a dedicated **pre-FoG model** (not just reusing the detection model's probabilities). Two variants:
- **3-class** version: Normal (0) / Pre-FoG (1, 2–5s before onset) / FoG (2)
- **2-class** version: Normal (0) / Any FoG signal — Pre-FoG or FoG combined (1)

Includes a dedicated `PreFoGDetector` model and `PreFoGDataset`, class-weight handling for the imbalance and evaluation of both variants.
---

# Part 2 - Brazil dataset


This part covers a new dataset recorded separately from FoG-STAR (5 IMU devices, video annotated in ELAN, participant IDs `FOGXXX`). The work here has two stages:

1. **Testing the FoG-STAR-trained model as-is** on the new data, to see how well it transfers.
2. **Building a labeled training set from the new data and training/validating a model directly on it**, using Leave-One-Subject-Out (LOSO) cross-validation.

## Stage 1 - Testing the FoG-STAR model on the Brazil data

These four notebooks form a linear pipeline: label → resample → format → run inference.

**`fog_labeling_from_eaf.ipynb`**
Labels all 5 IMU devices for **one** recording from its ELAN `.eaf` annotation file. Parses the EAF for `FOG` and `Task` tiers (start/end times in ms from video start), uses the first camera frame's LSL timestamp from the H5 file to convert EAF times into LSL time, then assigns `fog_label` and `task_label` to every IMU sample by timestamp overlap. Outputs one CSV per device with columns `timestamp_lsl | acc_x/y/z | quat_w/x/y/z | fog_label | task_label`, plus a visual verification plot. This is the single-recording version of the logic later automated in `pipeline_all_participants.ipynb`.

**`resample_imu.ipynb`**
Resamples the labeled CSVs from their original irregular sample rate to a fixed **60 Hz** (interpolating signals, majority-vote re-assignment of FoG labels per new sample), with a visual check comparing original vs. resampled signal and a sample-rate verification step. Includes an optional CSV-concatenation step at the end.

**`prepare_test_data.ipynb`**
Takes the resampled, labeled per-device CSVs and reformats them into the exact input shape the FoG-STAR model expects: aligns the three relevant devices (ankle L, ankle R, back), builds a combined dataframe in the required column order (`ankleL_q0-3 | ankleR_q0-3 | back_q0-3`), and segments into windows using the same relative-transform + mode-label logic as training. Saves `X_test_new.npy` / `y_test_new.npy`, with a value-range sanity check and a visual check of one FoG window.

**`inference_cnn.ipynb`**
Loads the trained `MultiBranchCNN` (`.pth` checkpoint from the FoG-STAR training) and runs inference on `X_test_new.npy`. Reports classification report, confusion matrix, ROC curve, and a sweep over decision thresholds — relevant here because the new data's FoG prevalence (~0.4%) is far lower than in training (~19.5%), so the default threshold is unlikely to be optimal.

## Stage 2 - New labeling pipeline and LOSO training on the Brazil data

**`pipeline_all_participants.ipynb`**
Automates the labeling step (Stage 1's `fog_labeling_from_eaf.ipynb` logic) across all participants and tasks at once, reading directly from OneDrive. For each (participant, task) pair: finds the matching H5 + EAF file pair via a task-name lookup table, parses the EAF, labels all 5 IMU devices via LSL synchronization, resamples to 100 Hz, and saves one CSV per device (`IMU_devX_FOGXXX_taskname_labeled_100Hz.csv`). Ends with a summary table of sample counts and FoG percentage per participant/device.

**`concatenate_tasks.ipynb`**
For each (subject, device) pair, concatenates all per-task labeled CSVs from `pipeline_all_participants.ipynb` into a single file (`IMU_devX_FOGXXX_all_tasks.csv`) and produces a summary (FoG % and task list per subject/device).

**`loso_training.ipynb`**
Trains and evaluates the `MultiBranchCNN` directly on the Brazil dataset using **Leave-One-Subject-Out cross-validation** (one fold per held-out subject), on **all tasks**. Requires manually filling in `DEVICE_MAPPING` (which physical device corresponds to `ankle_l` / `ankle_r` / `back` per subject, since device assignment wasn't consistent across recordings). Supports three input configurations via `INPUT_MODE`: `'acc'` (acceleration only, 9 channels), `'quat'` (quaternions only, 12 channels), or `'acc_quat'` (both, 21 channels). Includes FoG-window data augmentation (noise/reverse/scale, with selectable multiplier), per-fold and aggregated results, a results plot and per-task AUC breakdown.

**`loso_training_constrained.ipynb`**
Same LOSO setup and `MultiBranchCNN` architecture as `loso_training.ipynb`, but **restricted to two tasks** (`TASKS_TO_INCLUDE = ['8_Shape_Circuit_1', 'Narrow_Corridor_1']`) - the tasks richest in FoG events - rather than all tasks. Also goes further on the analysis side: post-processing (probability smoothing, minimum-duration filtering) tested on top of LOSO predictions, representative-subject plots (best/medium/worst fold), and PCA/t-SNE visualization of learned features.

