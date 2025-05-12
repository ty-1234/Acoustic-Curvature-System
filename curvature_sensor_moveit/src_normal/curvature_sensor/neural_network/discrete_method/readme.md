# Curvature Sensor Data Processing and Modeling Pipeline

This document outlines the pipeline I've set up for processing sensor data and training models, specifically focusing on a Gaussian Process Regressor (GPR) and a Random Forest Regressor for sanity checking.

## Pipeline Overview

The pipeline consists of three main Python scripts:

1.  `noise_remover.py`
2.  `RQ_GPR_train.py` (for GPR modeling)
3.  `randomForrest.py` (for Random Forest sanity check)

Here's how they work together:

### Step 1: Initial Data Cleaning with `noise_remover.py`

*   **Input**: My original, raw `merged_*.csv` files (located in `csv_data/merged/`).
*   **Process**: This script uses the `process_data_for_discrete_extraction` function. Its main responsibilities are:
    *   Identifying all distinct blocks where `Curvature_Active == 1`.
    *   For each active block:
        *   Deleting approximately 1 second of rows immediately *before* the block starts (to remove noisy transition data).
        *   Deleting approximately the last 1 second of rows *from within* the active block itself. If a block is very short (<=1s), this might remove the entire active segment.
    *   For the remaining parts of active blocks, it ensures the `Curvature` value is constant (taken from the original start of that block) and `Position_cm` is from the original data.
    *   Recalculating `Curvature_Active` based on the new `Curvature` values.
*   **Output**: The script saves the processed DataFrames into the `csv_data/cleaned/` directory. These "cleaned" files have had specific noisy segments and portions of active blocks removed.

### Step 2: Model Training and Evaluation

Both `RQ_GPR_train.py` and `randomForrest.py` operate on the same set of "cleaned" input files to ensure fair comparison.

*   **Input**: The `cleaned_*.csv` files from `csv_data/cleaned/` (the output of `noise_remover.py`). Both scripts are configured to read from this directory.
*   **Common Preprocessing (within both `RQ_GPR_train.py` and `randomForrest.py` for each cleaned file)**:
    *   **No Redundant Cleaning**: Critically, these scripts **do not** re-apply the `process_data_for_discrete_extraction` function. They assume the files are already cleaned regarding segment boundaries.
    *   **Idle Baseline Calculation**:
        *   For each "cleaned" file, an idle FFT baseline is calculated.
        *   This uses the logic: find the first row where `Position_cm == 0.0`, then take all *preceding* rows where `Curvature_Active == 0` and average their FFT values.
        *   *Important Note*: The success of this step relies on `noise_remover.py` leaving these initial idle rows intact. My understanding is that `noise_remover.py` primarily targets the end of an initial idle period just before activity, which should preserve the very early idle data needed for this baseline. I'll need to double-check the "cleaned" files to confirm this.
    *   **Filter for Active Data**: Only rows where `Curvature_Active == 1` are selected from the (already segment-cleaned) data.
    *   **Normalization**: The FFT features of these active rows are normalized by subtracting the calculated idle baseline.
    *   **No Downsampling**: I'm using all available data points for the `[test 1]` analysis.
    *   **Group ID for CV**: A group ID (the curvature value, e.g., "0.01", "0.05") is parsed from the filename. This is used for Leave-One-Curvature-Group-Out cross-validation.
*   **Model-Specific Steps**:
    *   **`RQ_GPR_train.py`**:
        *   Combines all processed segments.
        *   Performs Leave-One-Group-Out CV.
        *   Inside each fold: scales features (StandardScaler), defines the GPR kernel, trains the GPR, predicts, and evaluates (MSE, MAE, R²).
        *   Saves summary CV results, detailed predictions, and a final trained GPR model.
        *   Generates a summary markdown report.
    *   **`randomForrest.py` (Sanity Check)**:
        *   Follows the exact same common preprocessing steps as `RQ_GPR_train.py`.
        *   Performs Leave-One-Group-Out CV.
        *   Inside each fold: scales features, trains a `RandomForestRegressor` (e.g., `n_estimators=100`), predicts, and evaluates (MSE, MAE, R²).
        *   Saves its own summary CV results and detailed predictions, clearly marked as being from the Random Forest.
        *   Generates its own summary markdown report.

### Purpose of `randomForrest.py` as a Sanity Check

I've been seeing extremely poor performance with my GPR model (e.g., very large negative R² scores). This sanity check helps me figure out if the issue is:

1.  **Specific to the GPR model/setup**: GPRs can be tricky. If a simpler Random Forest performs significantly better on the exact same processed data, it points to issues with my GPR setup (kernel, optimization, bounds, etc.).
2.  **More fundamental to the data or features**: If the Random Forest *also* performs very poorly (as some previous runs indicated), it strongly suggests problems with the data itself. This could mean:
    *   The input features don't have enough predictive power for `Curvature`.
    *   The `Curvature` target has characteristics that make it hard for any model to learn.
    *   There might be an issue in earlier data processing stages (`noise_remover.py` or even data generation) impacting data quality.

The Random Forest provides a baseline. If it fails badly too, I'll focus on the data/feature pipeline. If it works reasonably well, I'll focus on debugging the GPR.

However this logic could be wrong since tree based models are not really able to interpolate data well.

### Current Status of Scripts

*   My `RQ_GPR_train.py` script has been updated to correctly omit the internal call to `process_data_for_discrete_extraction`. It now relies on the input files from `csv_data/cleaned/` being pre-processed by `noise_remover.py`. This avoids double-processing.
*   The overall pipeline (`noise_remover.py` followed by either the modified `RQ_GPR_train.py` or `randomForrest.py`) seems logical, provided `noise_remover.py` correctly prepares the data (especially preserving data for baseline calculation) and the subsequent scripts don't re-apply the same cleaning logic.

```// filepath: /Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/readme.md
# Curvature Sensor Data Processing and Modeling Pipeline

This document outlines the pipeline I've set up for processing sensor data and training models, specifically focusing on a Gaussian Process Regressor (GPR) and a Random Forest Regressor for sanity checking.

## Pipeline Overview

The pipeline consists of three main Python scripts:

1.  `noise_remover.py`
2.  `RQ_GPR_train.py` (for GPR modeling)
3.  `randomForrest.py` (for Random Forest sanity check)

Here's how they work together:

### Step 1: Initial Data Cleaning with `noise_remover.py`

*   **Input**: My original, raw `merged_*.csv` files (located in `csv_data/merged/`).
*   **Process**: This script uses the `process_data_for_discrete_extraction` function. Its main responsibilities are:
    *   Identifying all distinct blocks where `Curvature_Active == 1`.
    *   For each active block:
        *   Deleting approximately 1 second of rows immediately *before* the block starts (to remove noisy transition data).
        *   Deleting approximately the last 1 second of rows *from within* the active block itself. If a block is very short (<=1s), this might remove the entire active segment.
    *   For the remaining parts of active blocks, it ensures the `Curvature` value is constant (taken from the original start of that block) and `Position_cm` is from the original data.
    *   Recalculating `Curvature_Active` based on the new `Curvature` values.
*   **Output**: The script saves the processed DataFrames into the `csv_data/cleaned/` directory. These "cleaned" files have had specific noisy segments and portions of active blocks removed.

### Step 2: Model Training and Evaluation

Both `RQ_GPR_train.py` and `randomForrest.py` operate on the same set of "cleaned" input files to ensure fair comparison.

*   **Input**: The `cleaned_*.csv` files from `csv_data/cleaned/` (the output of `noise_remover.py`). Both scripts are configured to read from this directory.
*   **Common Preprocessing (within both `RQ_GPR_train.py` and `randomForrest.py` for each cleaned file)**:
    *   **No Redundant Cleaning**: Critically, these scripts **do not** re-apply the `process_data_for_discrete_extraction` function. They assume the files are already cleaned regarding segment boundaries.
    *   **Idle Baseline Calculation**:
        *   For each "cleaned" file, an idle FFT baseline is calculated.
        *   This uses the logic: find the first row where `Position_cm == 0.0`, then take all *preceding* rows where `Curvature_Active == 0` and average their FFT values.
        *   *Important Note*: The success of this step relies on `noise_remover.py` leaving these initial idle rows intact. My understanding is that `noise_remover.py` primarily targets the end of an initial idle period just before activity, which should preserve the very early idle data needed for this baseline. I'll need to double-check the "cleaned" files to confirm this.
    *   **Filter for Active Data**: Only rows where `Curvature_Active == 1` are selected from the (already segment-cleaned) data.
    *   **Normalization**: The FFT features of these active rows are normalized by subtracting the calculated idle baseline.
    *   **No Downsampling**: I'm using all available data points for the `[test 1]` analysis.
    *   **Group ID for CV**: A group ID (the curvature value, e.g., "0.01", "0.05") is parsed from the filename. This is used for Leave-One-Curvature-Group-Out cross-validation.
*   **Model-Specific Steps**:
    *   **`RQ_GPR_train.py`**:
        *   Combines all processed segments.
        *   Performs Leave-One-Group-Out CV.
        *   Inside each fold: scales features (StandardScaler), defines the GPR kernel, trains the GPR, predicts, and evaluates (MSE, MAE, R²).
        *   Saves summary CV results, detailed predictions, and a final trained GPR model.
        *   Generates a summary markdown report.
    *   **`randomForrest.py` (Sanity Check)**:
        *   Follows the exact same common preprocessing steps as `RQ_GPR_train.py`.
        *   Performs Leave-One-Group-Out CV.
        *   Inside each fold: scales features, trains a `RandomForestRegressor` (e.g., `n_estimators=100`), predicts, and evaluates (MSE, MAE, R²).
        *   Saves its own summary CV results and detailed predictions, clearly marked as being from the Random Forest.
        *   Generates its own summary markdown report.

### Purpose of `randomForrest.py` as a Sanity Check

I've been seeing extremely poor performance with my GPR model (e.g., very large negative R² scores). This sanity check helps me figure out if the issue is:

1.  **Specific to the GPR model/setup**: GPRs can be tricky. If a simpler Random Forest performs significantly better on the exact same processed data, it points to issues with my GPR setup (kernel, optimization, bounds, etc.).
2.  **More fundamental to the data or features**: If the Random Forest *also* performs very poorly (as some previous runs indicated), it strongly suggests problems with the data itself. This could mean:
    *   The input features don't have enough predictive power for `Curvature`.
    *   The `Curvature` target has characteristics that make it hard for any model to learn.
    *   There might be an issue in earlier data processing stages (`noise_remover.py` or even data generation) impacting data quality.

The Random Forest provides a baseline. If it fails badly too, I'll focus on the data/feature pipeline. If it works reasonably well, I'll focus on debugging the GPR.

### Current Status of Scripts

*   My `RQ_GPR_train.py` script has been updated to correctly omit the internal call to `process_data_for_discrete_extraction`. It now relies on the input files from `csv_data/cleaned/` being pre-processed by `noise_remover.py`. This avoids double-processing.
*   The overall pipeline (`noise_remover.py` followed by either the modified `RQ_GPR_train.py` or `randomForrest.py`) seems logical, provided `noise_remover.py` correctly prepares the data (especially preserving data for baseline calculation) and the subsequent scripts don't re-apply the same cleaning logic.
