# Random Forest (Sanity Check) Training Summary: Test Object `[test 1]`
**Report Generated:** 2025-05-12 17:15:33

## 1. Data Processing Overview
- **Target Test Object Identifier:** `[test 1]`
- **Number of Input CSV Files Scanned:** 5

### Normalization Baseline Calculation:
- Successfully calculated idle FFT baseline for **5** out of 5 scanned files.
- All scanned files had suitable baseline data or were processed accordingly (or skipped if empty/unreadable).

## 2. Cross-Validation Performance (Leave-One-Curvature-Out)
- **Number of CV Folds (Unique Curvature Groups):** 5
- **Average CV MSE:** 0.000570
- **Average CV MAE:** 0.018635
- **Average CV R² Score:** -207758937153353632492419584884736.0000

### Per-Fold Metrics:
| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Model Parameters (RF) |
|------|--------------|----------|----------|--------|------------------|-----------------------|
| 1 | 0.005 | 0.000542 | 0.019831 | -720468176188022315315424808402944.0000 | 0.22 | `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` |
| 2 | 0.007142 | 0.000162 | 0.010482 | -215573681139787948851133780328448.0000 | 0.22 | `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` |
| 3 | 0.01 | 0.000192 | 0.011408 | -63746027778188235275184780607488.0000 | 0.22 | `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` |
| 4 | 0.01818 | 0.000075 | 0.008134 | 0.0000 | 0.21 | `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` |
| 5 | 0.05 | 0.001878 | 0.043321 | -39006800660769775610345239347200.0000 | 0.21 | `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}` |

## 3. Generated Output Files
The following files were generated in or for the directory: `/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/discrete_method/model_outputs/RF_sanity_test1`
- `rf_sanity_test1_LOCO_cv_results.csv`
- `rf_sanity_test1_detailed_predictions.csv`
- `rf_sanity_test1_summary_report.md`