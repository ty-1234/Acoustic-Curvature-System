import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "..", "neural_network", "model_outputs", "idle_state_classifier")
test_data_path = os.path.join(script_dir, "data")  # UPDATE THIS

# === Load classifier model + scaler ===
model = joblib.load(os.path.join(model_dir, "idle_state_classifier.pkl"))
scaler = joblib.load(os.path.join(model_dir, "idle_state_scaler.pkl"))

# === Define feature set used by the idle state classifier ===
features_used = [
       "PC2", "PC1",
    "FFT_2000Hz", "FFT_200Hz", "FFT_1600Hz",
    "FFT_1800Hz", "FFT_1400Hz", "FFT_600Hz",
    "High_Band_Mean", "Mid_Band_Mean",
    "Low_Band_Mean", "Mid_to_Low_Band_Ratio"
]

# === Load and prepare test data ===
predictions = []
for fname in os.listdir(test_data_path):
    if not fname.endswith(".csv"):
        continue

    file_path = os.path.join(test_data_path, fname)
    df = pd.read_csv(file_path)
    
    # Check if required features exist
    missing_features = [f for f in features_used if f not in df.columns]
    if missing_features:
        print(f"⚠️ Skipping {fname}: Missing features: {missing_features}")
        continue
        
    if "Curvature_Active" not in df.columns:
        print(f"⚠️ Skipping {fname}: Missing 'Curvature_Active' column")
        continue

    X = df[features_used]
    y_true = df["Curvature_Active"].astype(int)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:,1]  # probability of active state
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"\n📄 {fname} → Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
    
    # Add predictions to dataframe
    df["Predicted_Active"] = y_pred
    df["Active_Probability"] = y_pred_proba
    
    # Save results
    output_csv = os.path.join(model_dir, f"active_predictions_{fname}")
    df.to_csv(output_csv, index=False)
    predictions.append((fname, accuracy, f1))

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Idle (0)', 'Active (1)'],
                yticklabels=['Idle (0)', 'Active (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{fname}\nAccuracy = {accuracy:.3f}, F1 = {f1:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"confusion_matrix_{fname.replace('.csv', '')}.png"))
    plt.close()
    
    # Save probability distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Active_Probability", hue="Curvature_Active", 
                 bins=50, kde=True, palette=["skyblue", "salmon"])
    plt.xlabel("Predicted Probability of Active State")
    plt.ylabel("Count")
    plt.title(f"{fname} - Active State Probability Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"probability_dist_{fname.replace('.csv', '')}.png"))
    plt.close()

# Final summary
print("\n=== Evaluation Summary ===")
for fname, accuracy, f1 in predictions:
    print(f"{fname}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

# Calculate overall metrics if multiple files were processed
if len(predictions) > 1:
    avg_accuracy = np.mean([acc for _, acc, _ in predictions])
    avg_f1 = np.mean([f1 for _, _, f1 in predictions])
    print(f"\nOverall: Accuracy = {avg_accuracy:.4f}, F1 = {avg_f1:.4f}")