"""
CSV Merger utility for curvature sensing data.

This script merges two types of CSV files containing data from the curvature sensing system:
1. Audio data (FFT features from acoustic signals)
2. Robot positional data

The files are merged based on timestamp alignment, creating a comprehensive dataset
that connects audio features with the robot's position during data collection.

Author: Bipindra Rai
Date: 2025-04-17
"""

import pandas as pd
import os


def merge_csvs(curvature_value):
    """
    Merges two CSV files (audio and robot data) based on the nearest timestamps.

    This function finds pairs of audio and robot CSV files with matching test identifiers,
    merges each pair based on the nearest timestamps, and saves the merged DataFrames
    to new CSV files in the 'csv_data/merged' directory.
    
    Parameters
    ----------
    curvature_value : float
        The curvature value used to identify the corresponding CSV files.
        This value should match the curvature used during data collection.
    
    Returns
    -------
    None
        The function does not return any value but creates merged CSV files in the
        output directory if successful.
    
    Notes
    -----
    The function expects the following file naming convention:
    - Audio files: raw_audio_{curvature_str}[test ID].csv
    - Robot files: raw_robot_{curvature_str}[test ID].csv
    
    The merged files will be named: merged_{curvature_str}[test ID].csv

    The [test ID] portion of filenames must be manually added to the raw data files
    before running this script. For example, to identify multiple tests at the same
    curvature, you might rename files to include identifiers like "[test 1]" or 
    "[session A]". If the test ID is missing, the script will attempt to match
    files based only on the curvature value.
    """

    # Get the base directory of the current script (curvature_sensor module root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Convert the curvature value to a string and replace the decimal point with an underscore
    curvature_str = str(curvature_value).replace(".", "_")
    
    # Directory paths
    raw_dir = os.path.join(base_dir, "csv_data", "raw")
    merged_dir = os.path.join(base_dir, "csv_data", "merged")
    
    # Ensure the merged directory exists
    os.makedirs(merged_dir, exist_ok=True)
    
    # Find all audio files matching the curvature value
    audio_pattern = f"raw_audio_{curvature_str}"
    audio_files = [f for f in os.listdir(raw_dir) if f.startswith(audio_pattern) and f.endswith(".csv")]
    
    if not audio_files:
        print(f"❌ No audio CSV files found with pattern: {audio_pattern}*.csv")
        return
    
    # Process each audio file
    for audio_file in audio_files:
        # Extract the test identifier if present (e.g., "[test 1]")
        test_id = ""
        if "[" in audio_file:
            test_id = audio_file[audio_file.index("["):audio_file.index("]")+1]
        
        # Construct the corresponding robot file name
        robot_file = f"raw_robot_{curvature_str}{test_id}.csv"
        robot_path = os.path.join(raw_dir, robot_file)
        audio_path = os.path.join(raw_dir, audio_file)
        
        # Check if the robot file exists
        if not os.path.exists(robot_path):
            print(f"❌ Robot CSV file not found: {robot_path}")
            continue
        
        print(f"🔄 Processing pair: {audio_file} and {robot_file}")
        
        # Construct the output file path
        output_filename = f"merged_{curvature_str}{test_id}.csv"
        merged_output_path = os.path.join(merged_dir, output_filename)
        
        # Load the audio and robot CSV files into pandas DataFrames
        print(f"📥 Loading CSV files...")
        df_audio = pd.read_csv(audio_path)  # Load the audio data
        df_robot = pd.read_csv(robot_path)  # Load the robot data

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

        # Save the merged DataFrame to the specified output path
        merged_df.to_csv(merged_output_path, index=False)

        # Print success messages with details about the output
        print(f"\n✅ Merge complete. Output saved to: {merged_output_path}")
        print(f"🧾 Rows in final CSV: {len(merged_df)}")


if __name__ == "__main__":
    """
    Entry point of the script.
    
    When executed directly, this script prompts the user to input a curvature value
    and then calls the merge_csvs function to process and merge the corresponding
    CSV files.
    """
    # Prompt the user to enter the curvature value
    curvature = float(input("Enter curvature value (e.g., 0.01818): "))
    
    # Call the merge_csvs function with the provided curvature value
    merge_csvs(curvature)
