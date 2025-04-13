import pandas as pd
import os

def merge_csvs(curvature_value):
    """
    Merges two CSV files (audio and robot data) based on the nearest timestamps.

    Parameters:
    - curvature_value (float): The curvature value used to identify the corresponding CSV files.

    The function:
    1. Constructs file paths for the audio and robot CSV files based on the curvature value.
    2. Checks if the required files exist.
    3. Loads the CSV files into pandas DataFrames.
    4. Merges the two DataFrames based on the nearest timestamps.
    5. Saves the merged DataFrame to a new CSV file in the 'csv_data/merged' directory.
    """

    # Get the base directory of the current script (curvature_sensor module root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Convert the curvature value to a string and replace the decimal point with an underscore
    curvature_str = str(curvature_value).replace(".", "_")
    
    # File paths for the input and output CSV files
    raw_audio_path = os.path.join(base_dir, "csv_data", "raw", f"raw_audio_{curvature_str}.csv")
    raw_robot_path = os.path.join(base_dir, "csv_data", "raw", f"raw_robot_{curvature_str}.csv")
    merged_output_path = os.path.join(base_dir, "csv_data", "merged", f"merged_{curvature_str}.csv")

    # Check if the audio CSV file exists
    if not os.path.exists(raw_audio_path):
        print(f"❌ Audio CSV file not found: {raw_audio_path}")
        return

    # Check if the robot CSV file exists
    if not os.path.exists(raw_robot_path):
        print(f"❌ Robot CSV file not found: {raw_robot_path}")
        return

    # Load the audio and robot CSV files into pandas DataFrames
    print("📥 Loading CSV files...")
    df_audio = pd.read_csv(raw_audio_path)  # Load the audio data
    df_robot = pd.read_csv(raw_robot_path)  # Load the robot data

    # Parse the 'Timestamp' column in both DataFrames into datetime objects for proper merging
    df_audio["Timestamp"] = pd.to_datetime(df_audio["Timestamp"])
    df_robot["Timestamp"] = pd.to_datetime(df_robot["Timestamp"])

    # Sort the DataFrames by the 'Timestamp' column to ensure proper alignment during merging
    df_audio.sort_values("Timestamp", inplace=True)
    df_robot.sort_values("Timestamp", inplace=True)

    print("🔄 Merging based on nearest timestamps...")

    # Perform the merge based on the nearest timestamp
    merged_df = pd.merge_asof(
        df_audio,  # Left DataFrame (audio data)
        df_robot,  # Right DataFrame (robot data)
        on="Timestamp",  # Column to merge on
        direction="nearest",  # Merge based on the nearest timestamp
        tolerance=pd.Timedelta(seconds=2)  # Maximum allowed difference between timestamps
    )

    # Ensure the 'csv_data/merged' directory exists before saving the merged file
    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)

    # Save the merged DataFrame to the specified output path
    merged_df.to_csv(merged_output_path, index=False)

    # Print success messages with details about the output
    print(f"\n✅ Merge complete. Output saved to: {merged_output_path}")
    print(f"🧾 Rows in final CSV: {len(merged_df)}")

if __name__ == "__main__":
    """
    Entry point of the script. Prompts the user to input a curvature value and calls the merge_csvs function.
    """
    # Prompt the user to enter the curvature value
    curvature = float(input("Enter curvature value (e.g., 0.01818): "))
    
    # Call the merge_csvs function with the provided curvature value
    merge_csvs(curvature)
