# GPR Model Training Summary: Test Object `[test 2]`
**Report Generated:** 2025-05-12 20:52:45

## 1. Data Processing Overview
- **Target Test Object Identifier:** `[test 2]`
- **Number of Input CSV Files Scanned:** 5

### Normalization Baseline Calculation:
- Successfully calculated idle FFT baseline for **5** out of 5 scanned files.
- All scanned files had suitable baseline data or were processed accordingly (or skipped if empty/unreadable).

## 2. Cross-Validation Performance (Leave-One-Curvature-Out)
- **Number of CV Folds (Unique Curvature Groups):** 5
- **Average CV MSE:** 0.000531
- **Average CV MAE:** 0.018145
- **Average CV R² Score:** -159380824665303802096589605961728.0000

### Per-Fold Metrics:
| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Learned Kernel |
|------|--------------|----------|----------|--------|------------------|----------------|
| 1 | 0.005 | 0.000541 | 0.021085 | -719245909150916397966327369695232.0000 | 426.71 | `1.09**2 * RationalQuadratic(alpha=0.481, length_scale=1.13) + WhiteKernel(noise_level=1e-05)` |
| 2 | 0.007142 | 0.000240 | 0.013023 | 0.0000 | 486.75 | `1.01**2 * RationalQuadratic(alpha=0.952, length_scale=1.03) + WhiteKernel(noise_level=1e-05)` |
| 3 | 0.01 | 0.000130 | 0.008957 | -43143759631626515870043449851904.0000 | 529.41 | `1.05**2 * RationalQuadratic(alpha=0.81, length_scale=1.02) + WhiteKernel(noise_level=1e-05)` |
| 4 | 0.01818 | 0.000081 | 0.006984 | 0.0000 | 434.59 | `1.07**2 * RationalQuadratic(alpha=0.746, length_scale=1.08) + WhiteKernel(noise_level=1e-05)` |
| 5 | 0.05 | 0.001662 | 0.040676 | -34514454543976123668174974484480.0000 | 471.07 | `1.06**2 * RationalQuadratic(alpha=1.14, length_scale=1.17) + WhiteKernel(noise_level=1e-05)` |

## 3. Final Model Information
- **Learned Kernel Parameters (Final Model):** `1.06**2 * RationalQuadratic(alpha=0.758, length_scale=0.997) + WhiteKernel(noise_level=1e-05)`
- **Final GPR Model Saved To:** `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test2/final_gpr_model_test2_data.joblib`
- **Final Scaler Saved To:** `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test2/final_scaler_test2_data.joblib`

## 4. Generated Output Files
The following files were generated in or for the directory: `/Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test2`
- `final_gpr_model_test2_data.joblib`
- `final_scaler_test2_data.joblib`
- `gpr_test2_LOCO_curvature_cv_results.csv`
- `gpr_test2_detailed_predictions.csv`
- `gpr_test2_summary_report.md`