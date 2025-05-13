## 📊 Data Processing & Modeling Pipeline Discrete Method

This document outlines the pipeline for processing raw sensor data and subsequently training and evaluating machine learning models to predict sensor curvature. The primary modeling script detailed here is `RQ_GPR_train.py` for Gaussian Process Regression. A Random Forest model is planned as a comparative benchmark.

Discrete Method:

### Pipeline Stages

**Stage 0: Raw Data Synchronization (via `csv_sync.py`)**
* **Input**: Raw acoustic sensor (FFT features) and robotic arm (position, clamp status) data.
    * Filenames: `raw_audio_<CURVATURE_STR>[TEST_SESSION_ID].csv` and `raw_robot_<CURVATURE_STR>[TEST_SESSION_ID].csv`.
* **Process**: `csv_sync.py` merges these streams by timestamp, annotating with `Curvature_Active`, `Position_cm`, and setting a nominal `Curvature` based on the experiment for active segments (0.0 otherwise).
* **Output**: `merged_<CURVATURE_STR>[TEST_SESSION_ID].csv` files in `csv_data/merged/`.
    * `<CURVATURE_STR>`: One of 5 nominal curvature profiles (e.g., `0_01`).
    * `[TEST_SESSION_ID]`: One of 4 repeated experimental runs (e.g., `[test 1]`).
    * Total: 20 `merged_*.csv` files (5 curvatures x 4 runs).

---

**Stage 1: Segment Cleaning & Noise Reduction (`noise_remover.py`)**
* **Input**: All 20 `merged_*.csv` files from `csv_data/merged/`.
* **Script**: `noise_remover.py` (using `process_data_for_discrete_extraction` function).
* **Key Operations**:
    * Identifies distinct active blocks (`Curvature_Active == 1`).
    * For each block: deletes ~1s of rows *before* its start and the last ~1s of rows *from within* the active block itself.
    * Standardizes `Curvature` (to its value at the original block start) and `Position_cm` for remaining active segments.
    * Recalculates `Curvature_Active`.
* **Output**: 20 `cleaned_<CURVATURE_STR>[TEST_SESSION_ID].csv` files in `csv_data/cleaned/`. These are the primary input for the modeling stage.

---

**Stage 2: GPR Model Training & Evaluation (`RQ_GPR_train.py`)**

This script focuses on training and evaluating a GPR model using data from a **single, operator-selected test session/run**.

* **Operator Input**: The `TARGET_TEST_SESSION_NUM` variable within `RQ_GPR_train.py` is set (e.g., to "1", "2", "3", or "4") to specify which test session's data to use.
* **File Selection**: The script filters files from `csv_data/cleaned/` to load only those matching the selected `TARGET_TEST_SESSION_NUM`. This results in **5 `cleaned_*.csv` files** being processed in a given run (one for each of the 5 curvature profiles, all from the chosen session).
* **Preprocessing for GPR (applied to each of the 5 selected cleaned files):**
    1.  **Idle Baseline Calculation**: For each input `cleaned_*.csv` file, an idle FFT baseline is calculated. This uses rows *from that same cleaned file* where `Curvature_Active == 0` AND that appear *before* the first row where `Position_cm == 0.0`. The FFTs from these rows are averaged. *(This assumes `noise_remover.py` preserves these initial idle rows).*
    2.  **Data Filtering**: Only rows with `Curvature_Active == 1` are selected for modeling.
    3.  **FFT Normalization**: The FFT features of these active rows are normalized by *subtracting* the file-specific idle baseline.
    4.  **No Data Downsampling**: All available active data points are used.
* **Data Aggregation**: Data from the 5 processed files (for the selected session) are combined.
* **Group ID for CV**: The `<CURVATURE_STR>` (e.g., "0.01", "0.05") is parsed from each of the 5 filenames. These 5 distinct curvature strings serve as group identifiers.
* **Cross-Validation Strategy (Leave-One-Curvature-Profile-Out):**
    * `LeaveOneGroupOut` CV is performed on the combined data from the selected test session.
    * In each of the 5 folds:
        * **Test Set**: Data corresponding to *one* curvature profile (i.e., from one of the 5 input files for that session).
        * **Training Set**: Data from the *other four* curvature profiles (from the other 4 input files of the same session).
* **GPR Model Training & Evaluation**:
    * Inside each CV fold: features are scaled (`StandardScaler`), a GPR model (ConstantKernel * RationalQuadratic + WhiteKernel) is trained, predictions are made, and performance is logged (MSE, MAE, R²).
* **Outputs (specific to the selected test session, e.g., for `[test 1]`):**
    * `gpr_test<N>_LOCO_curvature_cv_results.csv`: Summary CV metrics.
    * `gpr_test<N>_detailed_predictions.csv`: Per-instance prediction details.
    * Performance plots (Predicted vs. True, Residuals, etc.) saved as PNGs.
    * `final_gpr_model_test<N>_data.joblib` & `final_scaler_test<N>_data.joblib`: Final model and scaler trained on all data from the selected session.
    * (All outputs are saved in a session-specific subdirectory like `model_outputs/RQ_GPR_test<N>/`).

---

**Stage 3: Random Forest Sanity Check (`randomForrest.py`) - *Planned***

* *(This section can be filled in once the `randomForrest.py` script is developed. It should ideally follow the same data selection (operator chooses a test session) and preprocessing methodology as `RQ_GPR_train.py` to allow for a direct comparison of model performance on the same data subset.)*

This pipeline structure allows for modular data processing and focused model evaluation on specific experimental sessions, testing generalization across different curvature profiles within that session.




**discrete_method/mlp_model_outputs/Combined_Sessions_1_2_3_4_80_20_Split**

We performed the 80/20 random split as a sanity check to determine if the MLP could learn a basic relationship between the features (FFT + Position) and curvature when exposed to all curvature profiles during training. This helped isolate whether the poor performance in LOGO CV was due to the inherent difficulty of generalizing to unseen curvature profiles or a more fundamental issue with the features or model. The results showed that the MLP could learn weakly with proper preprocessing (R² ~0.018), confirming that the data contains some predictive signal and that preprocessing is crucial, but generalization to unseen profiles remains the primary challenge.

**discrete_method/mlp_model_outputs/Combined_Sessions_1_2_3_4_80_20_Split_NoNorm**
This was a control test where no normalization was applied during the 80/20 split to evaluate its impact on model performance. The results showed that normalization significantly improved the model, as the R² without normalization was approximately -0.0006, indicating slightly worse-than-random predictions.

