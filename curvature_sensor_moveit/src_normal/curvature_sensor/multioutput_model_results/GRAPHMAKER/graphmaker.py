import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
try:
    data = pd.read_csv('curvature_sensor_moveit/src_normal/curvature_sensor/multioutput_model_results/model_comparison.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'curvature_sensor_moveit/src_normal/curvature_sensor/multioutput_model_results/model_comparison.csv' was not found. Please check the file path.")

# Print column names for debugging
print("Available columns:", data.columns.tolist())

# Rename columns to match your code's expectations
data = data.rename(columns={
    'model': 'Model',
    'rmse_curvature_mean': 'RMSE_Curvature',
    'rmse_position_mean': 'RMSE_Position_cm'
})

# Sort models based on RMSE for consistent ordering
data_sorted_curvature = data.sort_values(by='RMSE_Curvature')
data_sorted_position = data.sort_values(by='RMSE_Position_cm')

# Find best (minimum) RMSE values for titles
best_curvature_rmse = data_sorted_curvature['RMSE_Curvature'].min()
best_position_rmse = data_sorted_position['RMSE_Position_cm'].min()

# Set bar colors for visual distinction
colors_curvature = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors_position = ['#17becf', '#bcbd22', '#9467bd', '#8c564b']

# Plot Curvature RMSE with detailed annotations
plt.figure(figsize=(10, 6))
bars_curvature = plt.bar(
    data_sorted_curvature['Model'], 
    data_sorted_curvature['RMSE_Curvature'], 
    color=colors_curvature[:len(data_sorted_curvature)], 
    alpha=0.8
)

# Annotate curvature RMSE values above bars
for bar in bars_curvature:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.0002, 
        f'{yval:.5f}', 
        ha='center', 
        va='bottom', 
        fontsize=10, 
        fontweight='bold'
    )

plt.title(f'Model Comparison - RMSE Curvature', fontsize=16, fontweight='bold')
plt.xlabel('Machine Learning Model', fontsize=14)
plt.ylabel('RMSE (Curvature, mm⁻¹)', fontsize=14)
plt.ylim(0, max(data_sorted_curvature['RMSE_Curvature']) + 0.002)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('detailed_rmse_curvature_comparison.png', dpi=300)
plt.show()

# Plot Position RMSE with detailed annotations
plt.figure(figsize=(10, 6))
bars_position = plt.bar(
    data_sorted_position['Model'], 
    data_sorted_position['RMSE_Position_cm'], 
    color=colors_position[:len(data_sorted_position)], 
    alpha=0.8
)

# Annotate position RMSE values above bars
for bar in bars_position:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.02, 
        f'{yval:.3f}', 
        ha='center', 
        va='bottom', 
        fontsize=10, 
        fontweight='bold'
    )

plt.title(f'Model Comparison - RMSE Position', fontsize=16, fontweight='bold')
plt.xlabel('Machine Learning Model', fontsize=14)
plt.ylabel('RMSE (Position, cm)', fontsize=14)
plt.ylim(0, max(data_sorted_position['RMSE_Position_cm']) + 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('detailed_rmse_position_comparison.png', dpi=300)
plt.show()