import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np 

# Define your FFT columns globally
FFT_COLUMNS = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 
               'FFT_1800Hz', 'FFT_2000Hz']
POSITION_COL = 'Position_cm'
CURVATURE_COL = 'Curvature'

def analyze_gpr_input_data(file_path, output_plot_dir):
    """
    Performs EDA on the data prepared for GPR input 
    (i.e., after loading all dev interpolated files, Curvature_Active==1 filter, and N-downsample).
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file for GPR input EDA ('{file_path}') not found.")
        print("Please ensure your main GPR training script saves the 'development_df_final_for_cv' DataFrame to this CSV file.")
        return

    base_name = os.path.basename(file_path)
    print(f"--- EDA for GPR Input Data from: {base_name} ---")
    
    try:
        df_gpr_input = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return

    if df_gpr_input.empty:
        print("Data file is empty. No EDA will be performed.")
        return

    print("\n--- Data Info ---")
    df_gpr_input.info()

    numerical_cols = [CURVATURE_COL, POSITION_COL] + [col for col in FFT_COLUMNS if col in df_gpr_input.columns]
    print("\n--- Basic Statistics (Numerical Columns) ---")
    print(df_gpr_input[numerical_cols].describe())

    # --- Target Variable: Curvature (Original Scale) ---
    print("\nGenerating Target Variable plots...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df_gpr_input[CURVATURE_COL], kde=True, bins=50)
    plt.title(f'Distribution of Target: {CURVATURE_COL}')
    plt.xlabel(CURVATURE_COL); plt.ylabel('Frequency'); plt.grid(True)
    plt.savefig(os.path.join(output_plot_dir, "01_gpr_input_curvature_dist.png"))
    plt.close()

    # --- Input Feature: Position_cm (Original Scale) ---
    if POSITION_COL in df_gpr_input.columns:
        print(f"Generating {POSITION_COL} plots...")
        fig_pos, axes_pos = plt.subplots(1, 2, figsize=(14, 5))
        position_counts = df_gpr_input[POSITION_COL].value_counts().sort_index()
        sns.barplot(x=position_counts.index, y=position_counts.values, color='skyblue', ax=axes_pos[0])
        axes_pos[0].set_title(f'Distribution of {POSITION_COL}')
        axes_pos[0].set_xlabel(POSITION_COL); axes_pos[0].set_ylabel('Frequency'); axes_pos[0].grid(axis='y')

        sns.scatterplot(x=df_gpr_input[POSITION_COL], y=df_gpr_input[CURVATURE_COL], alpha=0.2, s=10, ax=axes_pos[1])
        axes_pos[1].set_title(f'{POSITION_COL} vs. {CURVATURE_COL}')
        axes_pos[1].set_xlabel(POSITION_COL); axes_pos[1].set_ylabel(CURVATURE_COL); axes_pos[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "02_gpr_input_position_analysis.png"))
        plt.close(fig_pos)

    # --- Input Features: FFT Bands (as they are in the GPR input data file) ---
    print("\nGenerating FFT Distribution plots...")
    num_fft_cols = len(FFT_COLUMNS)
    n_rows_subplot = (num_fft_cols + 1) // 2 
    fig_fft_dist, axes_fft_dist = plt.subplots(n_rows_subplot, 2, figsize=(15, 3.5 * n_rows_subplot))
    axes_fft_dist = axes_fft_dist.flatten() # Flatten for easy iteration
    for i, fft_col in enumerate(FFT_COLUMNS):
        if fft_col in df_gpr_input.columns:
            sns.histplot(df_gpr_input[fft_col], kde=True, bins=30, ax=axes_fft_dist[i])
            axes_fft_dist[i].set_title(f'Distribution of {fft_col}')
            axes_fft_dist[i].grid(True)
        else:
            if i < len(axes_fft_dist): axes_fft_dist[i].axis('off') 
    # Hide any unused subplots if num_fft_cols is odd
    if num_fft_cols % 2 != 0 and len(axes_fft_dist) > num_fft_cols :
        axes_fft_dist[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, "03_gpr_input_fft_distributions.png"))
    plt.close(fig_fft_dist)

    print("Generating FFTs vs. Curvature (Scatter) plots...")
    fig_fft_scatter, axes_fft_scatter = plt.subplots(n_rows_subplot, 2, figsize=(15, 3.5 * n_rows_subplot))
    axes_fft_scatter = axes_fft_scatter.flatten()
    for i, fft_col in enumerate(FFT_COLUMNS):
        if fft_col in df_gpr_input.columns:
            sns.scatterplot(x=df_gpr_input[fft_col], y=df_gpr_input[CURVATURE_COL], alpha=0.1, s=5, ax=axes_fft_scatter[i])
            axes_fft_scatter[i].set_title(f'{fft_col} vs. {CURVATURE_COL}')
            axes_fft_scatter[i].set_xlabel(fft_col); axes_fft_scatter[i].set_ylabel(CURVATURE_COL); axes_fft_scatter[i].grid(True)
        else:
            if i < len(axes_fft_scatter): axes_fft_scatter[i].axis('off')
    if num_fft_cols % 2 != 0 and len(axes_fft_scatter) > num_fft_cols :
        axes_fft_scatter[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, "04_gpr_input_fft_vs_curvature.png"))
    plt.close(fig_fft_scatter)

    # --- Correlation Matrix for Input Features ---
    print("Generating Correlation Matrix...")
    input_features_for_corr = [col for col in FFT_COLUMNS if col in df_gpr_input.columns]
    if POSITION_COL in df_gpr_input.columns:
        input_features_for_corr.append(POSITION_COL)
    
    if input_features_for_corr:
        correlation_matrix = df_gpr_input[input_features_for_corr].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 7})
        plt.title('Correlation Matrix of Input Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "05_gpr_input_correlation_matrix.png"))
        plt.close()
        
    # --- Box Plots (Example: FFTs grouped by Position_cm) ---
    if POSITION_COL in df_gpr_input.columns and df_gpr_input[POSITION_COL].nunique() < 10 and df_gpr_input[POSITION_COL].nunique() > 1 : 
        print("Generating FFT Boxplots by Position_cm...")
        fft_subset_for_boxplot = FFT_COLUMNS[::max(1, len(FFT_COLUMNS)//3)] # Select a few

        fig_box, axes_box = plt.subplots(len(fft_subset_for_boxplot), 1, figsize=(10, 4 * len(fft_subset_for_boxplot)), sharex=True)
        if len(fft_subset_for_boxplot) == 1: # Handle if only one subplot
            axes_box = [axes_box] 
            
        for i, fft_col in enumerate(fft_subset_for_boxplot):
            if fft_col in df_gpr_input.columns:
                sns.boxplot(x=df_gpr_input[POSITION_COL], y=df_gpr_input[fft_col], ax=axes_box[i])
                axes_box[i].set_title(f'{fft_col} by {POSITION_COL}')
                axes_box[i].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "06_gpr_input_fft_boxplots_by_position.png"))
        plt.close(fig_box)
        
    print(f"--- Finished EDA for: {base_name} ---")
    print(f"Plots saved to: {output_plot_dir}")

if __name__ == "__main__":
    # === Configuration: Path to the data file prepared for GPR input ===
    # This file should be created by your main GPR training script by saving 
    # its 'development_df_final_for_cv' DataFrame.
    path_to_gpr_input_data_file = "temp_gpr_input_for_eda.csv" 
    
    # === Configuration: Directory to save plots ===
    output_plot_directory_main = "analysis_processed_gpr_input/" 
    if not os.path.exists(output_plot_directory_main):
        os.makedirs(output_plot_directory_main)
        print(f"Created plot directory: {output_plot_directory_main}")

    analyze_gpr_input_data(path_to_gpr_input_data_file, output_plot_directory_main)