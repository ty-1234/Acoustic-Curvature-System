"""
Model Comparison Visualization Script

This script generates comparison visualizations between ExtraTrees and 
Gaussian Process regression models for curvature prediction.

Author: Bipindra Rai
Date: April 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.ticker as mtick

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")

# Define colors for consistent visualization
ET_COLOR = '#1E3A8A'  # Navy blue for ExtraTrees
GPR_COLOR = '#B91C1C'  # Crimson for GPR

# Font sizes
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 12
ANNOTATION_SIZE = 12

# Directory paths - FIXED
base_dir = os.path.dirname(os.path.abspath(__file__))
et_dir = os.path.join(base_dir, "model_outputs", "extratrees", "BEST")
gpr_dir = os.path.join(base_dir, "model_outputs", "gpr_curvature")
output_dir = os.path.join(base_dir, "extra.vs.GPR")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def load_model_data():
    """Load and return all necessary CSV files for both models."""
    # Load ExtraTrees data
    et_predictions = pd.read_csv(os.path.join(et_dir, "et_prediction_data.csv"))
    et_metrics = pd.read_csv(os.path.join(et_dir, "et_performance_metrics.csv"))
    et_errors = pd.read_csv(os.path.join(et_dir, "et_error_distribution.csv"))
    
    # Load GPR data
    gpr_predictions = pd.read_csv(os.path.join(gpr_dir, "gpr_prediction_data.csv"))
    gpr_metrics = pd.read_csv(os.path.join(gpr_dir, "gpr_performance_metrics.csv"))
    gpr_errors = pd.read_csv(os.path.join(gpr_dir, "gpr_error_distribution.csv"))
    
    return {
        'et_pred': et_predictions,
        'et_metrics': et_metrics,
        'et_errors': et_errors,
        'gpr_pred': gpr_predictions,
        'gpr_metrics': gpr_metrics,
        'gpr_errors': gpr_errors
    }

def plot_true_vs_predicted(data):
    """
    Plot 1: True vs Predicted Curvature for both models with metrics
    Side-by-side comparison of ExtraTrees and GPR
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get data
    et_true = data['et_pred']['Curvature_True']
    et_predicted = data['et_pred']['Curvature_Predicted']
    gpr_true = data['gpr_pred']['Curvature_True']
    gpr_predicted = data['gpr_pred']['Curvature_Predicted']
    
    # Calculate metrics
    et_rmse = np.sqrt(mean_squared_error(et_true, et_predicted))
    et_r2 = r2_score(et_true, et_predicted)
    gpr_rmse = np.sqrt(mean_squared_error(gpr_true, gpr_predicted))
    gpr_r2 = r2_score(gpr_true, gpr_predicted)
    
    # Calculate polyfit for trend lines
    et_z = np.polyfit(et_true, et_predicted, 1)
    et_p = np.poly1d(et_z)
    
    gpr_z = np.polyfit(gpr_true, gpr_predicted, 1)
    gpr_p = np.poly1d(gpr_z)
    
    # Fixed max value for both plots (0.055 mm⁻¹)
    max_val = 0.055
    
    # ExtraTrees plot
    axes[0].scatter(et_true, et_predicted, alpha=0.6, color=ET_COLOR, edgecolor='none', s=30)
    axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Ideal (y=x)')
    
    # Add best fit line
    x_range = np.linspace(0, max_val, 100)
    axes[0].plot(x_range, et_p(x_range), color=ET_COLOR, linestyle='-', linewidth=1.5, 
                label=f'Best fit (y={et_z[0]:.3f}x+{et_z[1]:.4f})')
    
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_xlabel(r'True Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel(r'Predicted Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_title('ExtraTrees Regression', fontsize=TITLE_SIZE)
    axes[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    axes[0].annotate(f'RMSE = {et_rmse:.5f}\nR² = {et_r2:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=ANNOTATION_SIZE,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='lightgray'))
    axes[0].legend(fontsize=LEGEND_SIZE, loc='lower right')
    
    # GPR plot
    axes[1].scatter(gpr_true, gpr_predicted, alpha=0.6, color=GPR_COLOR, edgecolor='none', s=30)
    axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Ideal (y=x)')
    
    # Add best fit line
    axes[1].plot(x_range, gpr_p(x_range), color=GPR_COLOR, linestyle='-', linewidth=1.5,
                label=f'Best fit (y={gpr_z[0]:.3f}x+{gpr_z[1]:.4f})')
    
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].set_xlabel(r'True Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_ylabel(r'Predicted Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_title('Gaussian Process Regression', fontsize=TITLE_SIZE)
    axes[1].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    axes[1].annotate(f'RMSE = {gpr_rmse:.5f}\nR² = {gpr_r2:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=ANNOTATION_SIZE,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='lightgray'))
    axes[1].legend(fontsize=LEGEND_SIZE, loc='lower right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(data):
    """
    Plot 2: Residuals vs True Curvature for both models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get data and calculate residuals
    et_true = data['et_pred']['Curvature_True']
    et_residuals = data['et_pred']['Curvature_Predicted'] - data['et_pred']['Curvature_True']
    gpr_true = data['gpr_pred']['Curvature_True']
    gpr_residuals = data['gpr_pred']['Curvature_Predicted'] - data['gpr_pred']['Curvature_True']
    
    # Calculate standard deviation of residuals
    et_std = np.std(et_residuals)
    gpr_std = np.std(gpr_residuals)
    
    # Set max value for x-axis (consistent across both plots)
    max_val = 0.055
    
    # Fixed y-axis limits as requested [-0.04, 0.04]
    y_limits = (-0.04, 0.04)
    
    # ExtraTrees residuals
    axes[0].scatter(et_true, et_residuals, alpha=0.6, color=ET_COLOR, edgecolor='none', s=30)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(y_limits)
    axes[0].set_xlabel(r'True Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel('Residuals (Predicted - True)', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_title('ExtraTrees Residuals', fontsize=TITLE_SIZE)
    axes[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    axes[0].annotate(f'Std Dev = {et_std:.5f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=ANNOTATION_SIZE,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='lightgray'))
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # GPR residuals
    axes[1].scatter(gpr_true, gpr_residuals, alpha=0.6, color=GPR_COLOR, edgecolor='none', s=30)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(y_limits)
    axes[1].set_xlabel(r'True Curvature ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_ylabel('Residuals (Predicted - True)', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_title('Gaussian Process Residuals', fontsize=TITLE_SIZE)
    axes[1].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    axes[1].annotate(f'Std Dev = {gpr_std:.5f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=ANNOTATION_SIZE,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='lightgray'))
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_metrics(data):
    """
    Plot 3: Bar chart comparing performance metrics between models
    Split into two subplots: Error metrics and R² correlation coefficient
    """
    # Extract metrics from performance files
    et_metrics_df = data['et_metrics']
    gpr_metrics_df = data['gpr_metrics']
    
    # Set up metrics to compare - split into error metrics and R²
    error_metrics = ['RMSE', 'MAE', 'Max Error']
    correlation_metrics = ['R²']
    
    # Extract values by metric type
    et_error_values = et_metrics_df[et_metrics_df['Metric'].isin(error_metrics)]['Curvature'].values
    gpr_error_values = gpr_metrics_df[gpr_metrics_df['Metric'].isin(error_metrics)]['Curvature'].values
    
    et_r2_values = et_metrics_df[et_metrics_df['Metric'].isin(correlation_metrics)]['Curvature'].values
    gpr_r2_values = gpr_metrics_df[gpr_metrics_df['Metric'].isin(correlation_metrics)]['Curvature'].values
    
    # Create figure with two subplots stacked vertically, with more space for the error metrics
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1], 
                             sharex=False, gridspec_kw={'hspace': 0.3})
    
    # --- Top subplot: Error metrics (RMSE, MAE, Max Error) ---
    x_error = np.arange(len(error_metrics))
    width = 0.35
    
    # Error metrics bars
    bars1 = axes[0].bar(x_error - width/2, et_error_values, width, label='ExtraTrees', 
                        color=ET_COLOR, edgecolor='black', linewidth=0.5)
    bars2 = axes[0].bar(x_error + width/2, gpr_error_values, width, label='GPR', 
                        color=GPR_COLOR, edgecolor='black', linewidth=0.5)
    
    # Add horizontal baseline at y=0
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Styling for top subplot
    axes[0].set_ylabel(r'Error (mm$^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_title('Error Metrics', fontsize=TITLE_SIZE)
    axes[0].set_xticks(x_error)
    axes[0].set_xticklabels(error_metrics, fontsize=TICK_LABEL_SIZE)
    axes[0].tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[0].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Auto-scale y-axis with padding for error metrics
    axes[0].autoscale(axis='y')
    y_max = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, y_max * 1.15)  # Add 15% padding
    
    # Bar value labels for error metrics (horizontal text)
    def label_bars(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.5f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=ANNOTATION_SIZE-2)
    
    label_bars(bars1, axes[0])
    label_bars(bars2, axes[0])
    
    # --- Bottom subplot: R² score ---
    x_r2 = np.arange(len(correlation_metrics))
    
    # R² bars
    bars3 = axes[1].bar(x_r2 - width/2, et_r2_values, width, label='ExtraTrees', 
                        color=ET_COLOR, edgecolor='black', linewidth=0.5)
    bars4 = axes[1].bar(x_r2 + width/2, gpr_r2_values, width, label='GPR', 
                        color=GPR_COLOR, edgecolor='black', linewidth=0.5)
    
    # Add horizontal reference line at y=1 (perfect R²)
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Styling for bottom subplot
    axes[1].set_ylabel('R² Score', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_title('Correlation Metric', fontsize=TITLE_SIZE)
    axes[1].set_xticks(x_r2)
    axes[1].set_xticklabels(correlation_metrics, fontsize=TICK_LABEL_SIZE)
    axes[1].tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE)
    axes[1].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set appropriate y-axis limits for R²
    axes[1].set_ylim(0, 1.05)  # R² values are between 0 and 1
    
    # Bar value labels for R² (horizontal text)
    def label_r2_bars(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=ANNOTATION_SIZE)
    
    label_r2_bars(bars3, axes[1])
    label_r2_bars(bars4, axes[1])
    
    # Overall title for the entire figure
    fig.suptitle('Comparison of ExtraTrees vs GPR Model Performance', 
                fontsize=TITLE_SIZE+2, y=0.98)

    # First apply tight_layout without the rect parameter
    fig.tight_layout()

    # Then adjust the position of the suptitle if needed
    plt.subplots_adjust(top=0.9)  # Give 10% padding at the top for the suptitle

    plt.savefig(os.path.join(output_dir, 'comparison_metrics_split.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_histogram(data):
    """
    Plot 4: Step histogram of absolute prediction errors on log scale
    """
    # Extract absolute errors
    et_abs_errors = data['et_pred']['Curvature_AbsError']
    gpr_abs_errors = data['gpr_pred']['Curvature_AbsError']
    
    # Determine bin edges to ensure identical bins for both histograms
    max_error = max(et_abs_errors.max(), gpr_abs_errors.max())
    n_bins = 50  # Increased bins for more detailed steps
    bin_edges = np.linspace(0, max_error, n_bins+1)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Compute histograms (get bin counts)
    et_counts, _ = np.histogram(et_abs_errors, bins=bin_edges)
    gpr_counts, _ = np.histogram(gpr_abs_errors, bins=bin_edges)
    
    # Plot as step histograms
    plt.step(bin_edges[:-1], et_counts, where='post', color=ET_COLOR, 
             linewidth=2, label='ExtraTrees', solid_capstyle='round')
    plt.step(bin_edges[:-1], gpr_counts, where='post', color=GPR_COLOR, 
             linewidth=2, linestyle='--', label='GPR', solid_capstyle='round')
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Set x-axis limits to focus area
    plt.xlim(0, 0.01)
    
    # Apply nice labels and styling
    plt.xlabel(r'Absolute Prediction Error (mm$^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Log Count', fontsize=AXIS_LABEL_SIZE)
    plt.title('Comparison of Absolute Prediction Errors (Log Scale)', fontsize=TITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    
    # Minor grid lines for log scale
    plt.grid(which='minor', linestyle=':', alpha=0.4)
    
    # Set minimum y value to handle potential zeros in log scale
    plt.ylim(bottom=0.9)  # Start just below 1
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram_log.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_cdf(data):
    """
    Plot 5: Cumulative Distribution Function (CDF) of absolute errors
    With zoomed x-axis to show meaningful differences
    """
    # Extract absolute errors
    et_abs_errors = data['et_pred']['Curvature_AbsError'].sort_values().values
    gpr_abs_errors = data['gpr_pred']['Curvature_AbsError'].sort_values().values
    
    # Calculate cumulative percentages
    et_percentiles = np.arange(1, len(et_abs_errors) + 1) / len(et_abs_errors) * 100
    gpr_percentiles = np.arange(1, len(gpr_abs_errors) + 1) / len(gpr_abs_errors) * 100
    
    # Create CDF plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Main plots
    ax.plot(et_abs_errors, et_percentiles, color=ET_COLOR, label='ExtraTrees', linewidth=2)
    ax.plot(gpr_abs_errors, gpr_percentiles, color=GPR_COLOR, label='GPR', linewidth=2)
    
    # Find 90th percentile errors for annotation
    et_90th = np.percentile(et_abs_errors, 90)
    gpr_90th = np.percentile(gpr_abs_errors, 90)
    
    # Add horizontal and vertical reference lines
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=et_90th, color=ET_COLOR, linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=gpr_90th, color=GPR_COLOR, linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Zoom x-axis to [0, 0.02] range as requested
    ax.set_xlim(0, 0.02)
    
    # Clean annotations with curved arrows
    ax.annotate(f'ET 90%: {et_90th:.5f}', 
                xy=(et_90th, 90), 
                xytext=(et_90th - 0.001, 75),  # Position text below the line
                fontsize=ANNOTATION_SIZE,
                color=ET_COLOR,
                arrowprops=dict(arrowstyle='->', color=ET_COLOR, 
                                connectionstyle='arc3,rad=0.2', linewidth=1.5))
    
    ax.annotate(f'GPR 90%: {gpr_90th:.5f}', 
                xy=(gpr_90th, 90), 
                xytext=(gpr_90th + 0.001, 75),  # Position text below the line
                fontsize=ANNOTATION_SIZE,
                color=GPR_COLOR,
                arrowprops=dict(arrowstyle='->', color=GPR_COLOR, 
                                connectionstyle='arc3,rad=-0.2', linewidth=1.5))
    
    # Add grid with minor lines
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.grid(which='minor', linestyle=':', alpha=0.2)
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    # Set axis labels and title
    ax.set_xlabel(r'Absolute Error Threshold ($\mathrm{mm}^{-1}$)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=AXIS_LABEL_SIZE)
    ax.set_title('Cumulative Distribution of Prediction Errors', fontsize=TITLE_SIZE)
    
    # Format axes
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_cdf.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all visualization processes."""
    print("Loading model data files...")
    data = load_model_data()
    
    print("Generating comparison visualizations...")
    plot_true_vs_predicted(data)
    plot_residuals(data)
    plot_performance_metrics(data)
    plot_error_histogram(data)
    plot_error_cdf(data)
    
    print(f"All visualizations saved to: {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
