# GPR Model Training Summary: Test Object `[test 1]`
**Report Generated:** 2025-05-12 16:27:06

## 1. Data Processing Overview
- **Target Test Object Identifier:** `[test 1]`
- **Number of Input CSV Files Scanned:** 5

### Normalization Baseline Calculation:
- Successfully calculated idle FFT baseline for **5** out of 5 scanned files.
- All scanned files had suitable baseline data or were processed accordingly (or skipped if empty/unreadable).

## 2. Cross-Validation Performance (Leave-One-Curvature-Out)
- **Number of CV Folds (Unique Curvature Groups):** 5
- **Average CV MSE:** 0.000574
- **Average CV MAE:** 0.018748
- **Average CV R² Score:** -220567292232444744054737490411520.0000

### Per-Fold Metrics:
| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Learned Kernel |
|------|--------------|----------|----------|--------|------------------|----------------|
| 1 | 0.005 | 0.000509 | 0.019743 | -676758769039637705483060242284544.0000 | 262.91 | `0.844**2 * RationalQuadratic(alpha=1.27, length_scale=1.23) + WhiteKernel(noise_level=1e-05)` |
| 2 | 0.007142 | 0.000242 | 0.012530 | -321459523104952898769533314007040.0000 | 291.40 | `0.859**2 * RationalQuadratic(alpha=1.84, length_scale=1.15) + WhiteKernel(noise_level=1e-05)` |
| 3 | 0.01 | 0.000198 | 0.011202 | -65922724961499111666567841054720.0000 | 265.47 | `0.89**2 * RationalQuadratic(alpha=1.54, length_scale=1.2) + WhiteKernel(noise_level=1e-05)` |
| 4 | 0.01818 | 0.000060 | 0.007122 | 0.0000 | 322.04 | `0.891**2 * RationalQuadratic(alpha=1.63, length_scale=1.25) + WhiteKernel(noise_level=1e-05)` |
| 5 | 0.05 | 0.001863 | 0.043140 | -38695444056134089922918974750720.0000 | 279.72 | `1.03**2 * RationalQuadratic(alpha=0.316, length_scale=2.09) + WhiteKernel(noise_level=1e-05)` |

## 3. Final Model Information
- **Learned Kernel Parameters (Final Model):** `0.886**2 * RationalQuadratic(alpha=1.73, length_scale=1.14) + WhiteKernel(noise_level=1e-05)`
- **Final GPR Model Saved To:** `ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1/final_gpr_model_test1_data.joblib`
- **Final Scaler Saved To:** `ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1/final_scaler_test1_data.joblib`

## 4. Generated Output Files
The following files were generated in or for the directory: `ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RQ_GPR_test1`
- `final_gpr_model_test1_data.joblib`
- `final_scaler_test1_data.joblib`
- `gpr_test1_LOCO_curvature_cv_results.csv`
- `gpr_test1_detailed_predictions.csv`
- `gpr_test1_summary_report.md`