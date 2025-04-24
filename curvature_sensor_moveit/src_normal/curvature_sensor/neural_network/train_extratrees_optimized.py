import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Assuming X_train, y_train, X_test, y_test, output_dir are already defined

param_grid = {
    "estimator__n_estimators": [200, 400, 600, 800, 1000],
    "estimator__max_depth": [10, 20, 30, 40, 50, None],
    "estimator__min_samples_split": [2, 4, 6, 8],
    "estimator__min_samples_leaf": [1, 2, 4],
    "estimator__max_features": ["sqrt", "log2", None]
}

model = ExtraTreesRegressor(random_state=42)
search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, random_state=42, n_jobs=-1, cv=3, verbose=2)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"🔍 Best ExtraTrees parameters after search: {search.best_params_}")

# Create a unique output directory for the final model and results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_output_dir = os.path.join(output_dir, f"extratrees_final_{timestamp}")
os.makedirs(model_output_dir, exist_ok=True)

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model
model_path = os.path.join(model_output_dir, "extratrees_model.joblib")
joblib.dump(best_model, model_path)

# Save the metrics
metrics_path = os.path.join(model_output_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"Mean Absolute Error: {mae}\n")
    f.write(f"R^2 Score: {r2}\n")

# Plot and save feature importances
plt.figure(figsize=(10, 6))
importances = best_model.feature_importances_
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.savefig(os.path.join(model_output_dir, "feature_importances.png"))
plt.close()

# Plot and save predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig(os.path.join(model_output_dir, "predicted_vs_actual.png"))
plt.close()
