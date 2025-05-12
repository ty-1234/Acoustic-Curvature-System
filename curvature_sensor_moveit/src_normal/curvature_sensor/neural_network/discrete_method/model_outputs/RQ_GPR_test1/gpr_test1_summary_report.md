# GPR Model Training Summary: Test Object `[test 1]`
**Report Generated:** 2025-05-12 19:37:35

## 1. Data Processing Overview
- **Target Test Object Identifier:** `[test 1]`
- **Number of Input CSV Files Scanned:** 5

### Normalization Baseline Calculation:
- Successfully calculated idle FFT baseline for **5** out of 5 scanned files.
- All scanned files had suitable baseline data or were processed accordingly (or skipped if empty/unreadable).

## 2. Cross-Validation Performance (Leave-One-Curvature-Out)
- **Number of CV Folds (Unique Curvature Groups):** 5
- **Average CV MSE:** 0.000582
- **Average CV MAE:** 0.019201
- **Average CV R² Score:** -150904893072788318533031607402496.0000

### Per-Fold Metrics:
| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Learned Kernel |
|------|--------------|----------|----------|--------|------------------|----------------|
| 1 | 0.005 | 0.000520 | 0.020032 | -690642257592447403228361547317248.0000 | 508.84 | `1.05**2 * RationalQuadratic(alpha=0.464, length_scale=1.12) + WhiteKernel(noise_level=1e-05)` |
| 2 | 0.007142 | 0.000265 | 0.013368 | 0.0000 | 450.59 | `1.03**2 * RationalQuadratic(alpha=0.675, length_scale=1.04) + WhiteKernel(noise_level=1e-05)` |
| 3 | 0.01 | 0.000165 | 0.010487 | -54995973680982639535286554460160.0000 | 438.76 | `1.05**2 * RationalQuadratic(alpha=0.684, length_scale=1.03) + WhiteKernel(noise_level=1e-05)` |
| 4 | 0.01818 | 0.000107 | 0.009145 | -8886234090511563412308817346560.0000 | 417.88 | `1.08**2 * RationalQuadratic(alpha=0.606, length_scale=1.05) + WhiteKernel(noise_level=1e-05)` |
| 5 | 0.05 | 0.001851 | 0.042971 | 0.0000 | 430.75 | `1.11**2 * RationalQuadratic(alpha=0.389, length_scale=1.67) + WhiteKernel(noise_level=1e-05)` |

## 3. Final Model Information
- **Learned Kernel Parameters (Final Model):** `1.07**2 * RationalQuadratic(alpha=0.621, length_scale=0.999) + WhiteKernel(noise_level=1e-05)`
- **Final GPR Model Saved To:** `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1/final_gpr_model_test1_data.joblib`
- **Final Scaler Saved To:** `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1/final_scaler_test1_data.joblib`

## 4. Generated Output Files
The following files were generated in or for the directory: `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1`
- `final_gpr_model_test1_data.joblib`
- `final_scaler_test1_data.joblib`
- `gpr_test1_LOCO_curvature_cv_results.csv`
- `gpr_test1_detailed_predictions.csv`
- `gpr_test1_summary_report.md`