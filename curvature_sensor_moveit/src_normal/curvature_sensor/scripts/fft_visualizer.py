import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def list_csv_files(directory):
    """
    Lists all CSV files in the specified directory.

    Parameters:
        directory (str): Path to the directory to scan for CSV files.

    Returns:
        list: List of CSV file paths.
    """
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []

    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No CSV files found in directory: {directory}")
    return csv_files

def plot_fft_by_section(csv_path, kind='box'):
    """
    Visualizes FFT amplitudes grouped by robot sections.

    Parameters:
        csv_path (str): Path to the merged CSV file.
        kind (str): 'box' for boxplot (default), 'line' for time series, 'average' for mean FFT, 'heatmap' for heatmap.

    Returns:
        None
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    if not fft_cols:
        print("⚠️ No FFT columns found.")
        return

    if kind in ['box', 'average'] and "Section" not in df.columns:
        print("⚠️ Missing 'Section' column for this plot type.")
        return

    print(f"📂 Loaded: {csv_path}")
    print(f"🔢 FFT columns: {fft_cols}")

    try:
        if kind == 'box':
            # 📊 Boxplot: Used to compare FFT amplitude distributions across sections
            # This type of plot is useful for visualizing the spread of FFT amplitudes
            # across different sections of the sensor.
            sections = df["Section"].unique()
            plt.figure(figsize=(12, 6))
            for section in sections:
                section_data = df[df["Section"] == section][fft_cols]
                plt.boxplot(section_data.values, positions=np.arange(len(fft_cols)) + len(fft_cols) * sections.tolist().index(section),
                            widths=0.6, patch_artist=True, boxprops=dict(facecolor=f"C{sections.tolist().index(section)}"),
                            labels=fft_cols if sections.tolist().index(section) == 0 else None)
            plt.title("FFT Amplitude Distributions by Section")
            plt.xlabel("Frequency")
            plt.ylabel("Amplitude")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif kind == 'line':
            # 📈 Line Plot: Used to compare FFT curves across curvatures
            # This type of plot shows how FFT amplitudes change over time or across frequency bands.
            if "Timestamp" not in df.columns:
                print("⚠️ Missing 'Timestamp' column for line plot.")
                return
            plt.figure(figsize=(12, 6))
            for freq in fft_cols:
                plt.plot(df['Timestamp'], df[freq], label=freq)
            plt.title("FFT Amplitudes Over Time")
            plt.xlabel("Timestamp")
            plt.ylabel("Amplitude")
            plt.xticks(rotation=45)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.show()

        elif kind == 'average':
            # 📊 Mean FFT Plot: Used to show average FFT amplitudes per section
            # This type of plot is useful for summarizing how different sections respond
            # to specific frequency bands.
            mean_df = df.groupby("Section")[fft_cols].mean().T
            mean_df.index = [int(f.replace("FFT_", "").replace("Hz", "")) for f in mean_df.index]
            mean_df.sort_index(inplace=True)
            plt.figure(figsize=(10, 6))
            for section in mean_df.columns:
                plt.plot(mean_df.index, mean_df[section], marker='o', label=f"Section {section}")
            plt.title("Mean FFT Amplitudes by Section")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Mean Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()

        elif kind == 'heatmap':
            # 🔥 Heatmap: Used to visualize FFT amplitudes over time
            # This type of plot helps identify sudden energy shifts or patterns
            # across frequency bands and time.
            fft_only = df[fft_cols]
            plt.figure(figsize=(12, 6))
            plt.imshow(fft_only.T, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Amplitude')
            plt.title("FFT Amplitudes Over Time")
            plt.ylabel("Frequency Index")
            plt.xlabel("Sample Index")
            plt.tight_layout()
            plt.show()

        else:
            print("❌ Invalid plot type. Use 'box', 'line', 'average', or 'heatmap'.")

    except Exception as e:
        print(f"❌ Error generating plot: {e}")

def main():
    """
    Main function for the FFT Visualizer.
    Allows the user to select a CSV file and visualization type.

    📊 Plots Used in the Original Paper:
    1. Bar Graphs of FFT Amplitudes by Frequency Band:
       - X-axis: Frequency band (e.g., 200 Hz to 2000 Hz in 200 Hz steps)
       - Y-axis: Normalized FFT amplitude
       - ✅ Used to illustrate how different frequency bands respond to a specific deformation.

    2. Line Graphs Comparing FFT Curves Across Curvatures:
       - Multiple lines, each corresponding to a curvature condition
       - ✅ Used to show how bending the sensor causes distinct FFT patterns.

    3. Scatter Plots or Regression Plots:
       - Curvature (X-axis) vs. FFT amplitude or model output (Y-axis)
       - ✅ Used to evaluate regression performance.

    🧠 Insight:
    - As curvature increases, FFT amplitudes at certain frequencies either increase or decrease noticeably.
    - Visualizations in the paper were primarily:
      - Static per-curvature bar/line plots
      - Overlaid FFT line plots per section or curvature
      - Model fit plots (regression accuracy)
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(base_dir, "csv_data", "merged")

    csv_files = list_csv_files(csv_dir)
    if not csv_files:
        exit()

    print("\nAvailable Merged CSVs:")
    for idx, file in enumerate(csv_files, 1):
        print(f"{idx}. {file}")

    try:
        file_choice = int(input("Select a file number to visualize: ")) - 1
        if file_choice not in range(len(csv_files)):
            print("❌ Invalid selection.")
            exit()
    except ValueError:
        print("❌ Invalid input.")
        exit()

    csv_path = os.path.join(csv_dir, csv_files[file_choice])

    print("\nChoose a visualization type:")
    print("box     → Boxplot by section (compare spread)")
    print("line    → Line plot over time (debug section transitions)")
    print("average → Mean FFT per section (used in the paper)")
    print("heatmap → Heatmap of FFT over time (identify sudden energy shifts)")

    plot_kind = input("Enter plot type: ").strip().lower()
    plot_fft_by_section(csv_path, kind=plot_kind)

if __name__ == "__main__":
    main()
