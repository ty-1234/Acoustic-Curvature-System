import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def analyze_single_raw_merged_file(file_path, output_plot_dir_individual):
    """
    Performs EDA on a single raw merged CSV file.
    Generates and saves time series plots and basic distributions.
    """
    base_name = os.path.basename(file_path)
    print(f"--- EDA for Raw Merged File: {base_name} ---")
    
    try:
        df_raw = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return

    if df_raw.empty:
        print(f"File {base_name} is empty. Skipping.")
        return

    # Define columns to plot
    curvature_col = 'Curvature'
    activity_col = 'Curvature_Active' # Original activity flag
    position_col = 'Position_cm'
    timestamp_col = 'Timestamp' 
    fft_cols = [f'FFT_{hz}Hz' for hz in range(200, 2001, 200)]
    fft_cols_to_plot = [col for col in [fft_cols[0], fft_cols[len(fft_cols)//2], fft_cols[-1]] if col in df_raw.columns]


    time_axis = df_raw.index 
    time_label = 'Row Index (Time)'
    
    if 'time_seconds' in df_raw.columns: # If your merger script creates this
        time_axis = df_raw['time_seconds']
        time_label = 'Time (seconds)'
    elif timestamp_col in df_raw.columns:
        try:
            df_raw['timestamp_dt_eda'] = pd.to_datetime(df_raw[timestamp_col])
            time_axis = (df_raw['timestamp_dt_eda'] - df_raw['timestamp_dt_eda'].iloc[0]).dt.total_seconds()
            time_label = 'Time (seconds from start)'
        except Exception:
            pass # Stick to row index if conversion fails

    # --- Time Series Plots ---
    print(f"  Generating Time Series plots for {base_name}...")
    fig_ts, axes_ts = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

    axes_ts[0].plot(time_axis, df_raw[curvature_col], label=curvature_col, color='blue')
    axes_ts[0].set_title(f'{curvature_col} Over Time (Raw Merged: {base_name})')
    axes_ts[0].set_ylabel(curvature_col)
    axes_ts[0].legend()
    axes_ts[0].grid(True)

    if activity_col in df_raw.columns:
        axes_ts[1].plot(time_axis, df_raw[activity_col], label=activity_col, color='orange', linestyle='-')
        axes_ts[1].set_title(f'{activity_col} Over Time (Raw Merged)')
        axes_ts[1].set_ylabel(activity_col)
        axes_ts[1].set_yticks([0, 1])
        axes_ts[1].legend()
        axes_ts[1].grid(True)
    else:
        axes_ts[1].text(0.5, 0.5, f'{activity_col} not found', ha='center', va='center')


    if position_col in df_raw.columns:
        axes_ts[2].plot(time_axis, df_raw[position_col], label=position_col, color='green', marker='.', linestyle='None')
        axes_ts[2].set_title(f'{position_col} Over Time (Raw Merged - NaNs show gaps)')
        axes_ts[2].set_ylabel(position_col)
        axes_ts[2].legend()
        axes_ts[2].grid(True)
    else:
        axes_ts[2].text(0.5, 0.5, f'{position_col} not found', ha='center', va='center')


    if fft_cols_to_plot:
        for fft_col in fft_cols_to_plot:
            if fft_col in df_raw.columns:
                axes_ts[3].plot(time_axis, df_raw[fft_col], label=fft_col, alpha=0.8)
        axes_ts[3].set_title('Selected Raw FFT Bands Over Time (Raw Merged)')
        axes_ts[3].set_ylabel('FFT Amplitude')
        axes_ts[3].legend()
        axes_ts[3].grid(True)
    else:
        axes_ts[3].text(0.5, 0.5, 'Selected FFT columns not found', ha='center', va='center')

    axes_ts[3].set_xlabel(time_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir_individual, f"{base_name.replace('.csv','')}_timeseries.png"))
    plt.close(fig_ts) 

    # --- Distribution Plots ---
    print(f"  Generating Distribution plots for {base_name}...")
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df_raw[curvature_col], kde=True, ax=axes_dist[0], bins=30)
    axes_dist[0].set_title(f'Distribution of {curvature_col}')
    axes_dist[0].grid(True)

    if position_col in df_raw.columns:
        sns.histplot(df_raw[position_col].dropna(), kde=False, 
                     discrete=True if df_raw[position_col].nunique() < 15 else False, 
                     bins=df_raw[position_col].nunique() if df_raw[position_col].nunique() < 15 else 30,
                     ax=axes_dist[1])
        axes_dist[1].set_title(f'Distribution of {position_col}')
        axes_dist[1].grid(True)
    else:
        axes_dist[1].text(0.5, 0.5, f'{position_col} not found', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir_individual, f"{base_name.replace('.csv','')}_distributions.png"))
    plt.close(fig_dist)
    print(f"--- Finished EDA for: {base_name}. Plots saved. ---")

if __name__ == "__main__":
    # === Configuration: Directory containing your raw "merged_*.csv" files ===
    raw_data_input_directory = "../csv_data/merged/" # Input path
    
    # === Configuration: Directory to save plots ===
    output_plot_directory_main = "analysis_output/raw/" # Output path
    
    # Ensure the output directory exists
    if not os.path.exists(output_plot_directory_main):
        os.makedirs(output_plot_directory_main)
        print(f"Created plot directory: {output_plot_directory_main}")

    raw_csv_files = sorted(glob.glob(os.path.join(raw_data_input_directory, "merged_*.csv")))

    if not raw_csv_files:
        print(f"No 'merged_*.csv' files found in '{raw_data_input_directory}'. Please check the path.")
    else:
        print(f"Found {len(raw_csv_files)} raw merged files to analyze.")
        for file_to_analyze in raw_csv_files:
            analyze_single_raw_merged_file(file_to_analyze, output_plot_directory_main)
        print(f"\nFinished processing all {len(raw_csv_files)} raw merged files for EDA.")
        print(f"Plots saved in: {output_plot_directory_main}")