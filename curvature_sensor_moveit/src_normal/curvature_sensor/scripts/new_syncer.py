"""
CSV Merger utility for curvature sensing data.

This script merges two types of CSV files containing data from the curvature sensing system:
1. Audio data (FFT features from acoustic signals)
2. Robot positional data

KEY TECHNICAL IMPLEMENTATION DETAILS:

Timestamp Synchronization:
------------------------
The script handles a critical timestamp synchronization issue between two data sources:
- Audio data uses UTC timestamps (created using pd.Timestamp.now(tz='UTC'))
- Robot data uses local system time (created using datetime.datetime.now())

This creates a consistent offset (typically 1 hour in UTC+1 timezones) between otherwise 
simultaneous recordings. The solution:
1. Detect timestamp offset by calculating the difference between earliest timestamps
2. If offset is approximately 1 hour (0.9-1.1 hours), adjust robot timestamps by subtracting
   this difference
3. Store original timestamps in "Original_Timestamp" column for reference
4. Use adjusted timestamps for all interval matching operations

Curvature Activation Detection:
-----------------------------
The script identifies intervals where the curvature sensor is actively measuring:
1. Find all robot events where Clamp="close" (handles various capitalizations if needed)
2. For each "close" event, find the next corresponding "open" event
3. Create intervals between each close-open pair, recording robot position data
4. Mark all audio data points falling within these intervals as Curvature_Active=1
   and assign the correct Position_cm value
5. Rows outside these intervals are marked Curvature_Active=0 with Curvature=0.0

Output Dataset:
-------------
The resulting merged CSV contains:
- All original audio FFT features
- Curvature value from the filename (e.g., 0.01818)
- Curvature_Active flag (1=active measurement, 0=inactive)
- Position_cm values propagated from robot data during active intervals

This approach enables precise alignment of sensor data with robot positioning,
despite the timezone differences in the original recordings.

Author: Bipindra Rai
Date: 2025-04-17
"""

import pandas as pd
import os
import numpy as np
from datetime import timedelta
import shutil  # Add this import at the top of the file
import re  # Add regex support


def get_all_curvature_values(raw_dir):
    """
    Extracts all unique curvature values from the raw data files.
    
    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw data files.
        
    Returns
    -------
    list
        List of unique curvature values found in the filenames.
    """
    audio_fft_dir = os.path.join(raw_dir, "audio_fft")
    robot_dir = os.path.join(raw_dir, "robot")
    curvature_values = set()
    
    # Check if directories exist
    if os.path.exists(audio_fft_dir):
        audio_files = [f for f in os.listdir(audio_fft_dir) if f.endswith('.csv')]
        for file in audio_files:
            if file.startswith('raw_audio_'):
                # Extract the curvature string (between underscore and bracket or end)
                parts = file.split('_', 2)  # Split into ['raw', 'audio', 'curvature+testid']
                if len(parts) >= 3:
                    curvature_str = parts[2]
                    # Remove test ID if present
                    if '[' in curvature_str:
                        curvature_str = curvature_str[:curvature_str.index('[')]
                    # Convert underscore-formatted curvature back to float
                    try:
                        curvature_value = float(curvature_str.replace('_', '.'))
                        curvature_values.add(curvature_value)
                    except ValueError:
                        # Skip if conversion fails
                        continue
    
    # Check robot directory as well
    if os.path.exists(robot_dir):
        robot_files = [f for f in os.listdir(robot_dir) if f.endswith('.csv')]
        for file in robot_files:
            if file.startswith('raw_robot_'):
                parts = file.split('_', 2)
                if len(parts) >= 3:
                    curvature_str = parts[2]
                    if '[' in curvature_str:
                        curvature_str = curvature_str[:curvature_str.index('[')]
                    try:
                        curvature_value = float(curvature_str.replace('_', '.'))
                        curvature_values.add(curvature_value)
                    except ValueError:
                        continue
    
    return sorted(list(curvature_values))

# Global list to track reasons for moving files to bad_data
bad_file_reasons = []

# Global list to track successfully merged files
merged_summary = []

# New list to track processed pairs
processed_pairs = []

# Add this new global list to track unpaired files
unpaired_files = []

def merge_csvs(curvature_value):
    """
    Merges two CSV files (audio and robot data) based on curvature activation intervals.

    This function finds pairs of audio and robot CSV files with matching test identifiers,
    identifies intervals where the robot clamp is active (between "close" and "open" states),
    and marks audio data within those intervals with the corresponding Section and 
    Curvature_Active flag.
    
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
    """

    # Get the base directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Convert the curvature value to a string and replace the decimal point with an underscore
    curvature_str = str(curvature_value).replace(".", "_")
    
    # Directory paths - using relative paths to csv_data directory
    raw_dir = os.path.join(script_dir, "..", "csv_data", "raw")
    merged_dir = os.path.join(script_dir, "..", "csv_data", "merged")
    
    # Create no_pairs directory
    no_pairs_dir = os.path.join(raw_dir, "no_pairs")
    os.makedirs(no_pairs_dir, exist_ok=True)
    
    # Ensure the merged directory exists
    os.makedirs(merged_dir, exist_ok=True)
    
    # Find all audio files matching the curvature value
    audio_pattern = f"raw_audio_{curvature_str}"
    audio_fft_dir = os.path.join(raw_dir, "audio_fft")
    
    # More precise matching to avoid selecting files with different curvature values
    audio_files = []
    for f in os.listdir(audio_fft_dir):
        if not f.endswith(".csv"):
            continue
        
        # Check if it starts with the audio pattern
        if not f.startswith(audio_pattern):
            continue
            
        # Make sure there's either nothing after the curvature string
        # or it starts with a test identifier (starts with "[")
        remaining = f[len(audio_pattern):]
        if not remaining or remaining.startswith("["):
            audio_files.append(f)
    
    if not audio_files:
        print(f"❌ No audio CSV files found with pattern: {audio_pattern}*.csv")
        return
    
    # Process each audio file
    for audio_file in audio_files:
        # Extract the test identifier if present (e.g., "[test 1]")
        test_id = ""
        try:
            print(f"🔍 Processing audio file: {audio_file}")  # Debugging statement
            if "[" in audio_file and "]" in audio_file:
                test_id = audio_file[audio_file.index("["):audio_file.index("]")+1]
            else:
                raise ValueError(f"Missing brackets in filename: {audio_file}")
        except ValueError as e:
            print(f"⚠️ Error processing file: {audio_file}")
            print(f"   Reason: {e}")
            test_id = ""
        
        # Find matching robot files based on curvature and test ID
        robot_dir = os.path.join(raw_dir, "robot")
        
        # Simplified logging - just show we're looking for a match
        print(f"🔐 Looking for exact match: raw_robot_{curvature_str}{test_id}.csv")
        
        matching_robot_files = [
            f for f in os.listdir(robot_dir)
            if f == f"raw_robot_{curvature_str}{test_id}.csv"
        ]
        
        if not matching_robot_files:
            print(f"❌ No matching robot file found for audio file: {audio_file}")
            print(f"   Expected robot file: raw_robot_{curvature_str}{test_id}.csv")
            print(f"   Available robot files: {os.listdir(robot_dir)}")  # Log all robot files

            # Move the unpaired audio file to no_pairs directory
            audio_path = os.path.join(audio_fft_dir, audio_file)
            shutil.move(audio_path, os.path.join(no_pairs_dir, audio_file))

            # Track this unpaired file
            unpaired_files.append({
                "file": audio_file,
                "type": "audio",
                "reason": "No matching robot file"
            })

            print(f"✓ Moved unpaired audio file to {no_pairs_dir}/{audio_file}")
            continue
            
        # Use the first matching robot file
        robot_file = matching_robot_files[0]
        robot_path = os.path.join(robot_dir, robot_file)
        audio_path = os.path.join(audio_fft_dir, audio_file)
        
        # Track this processed pair
        processed_pairs.append({
            "audio_file": audio_file,
            "robot_file": robot_file
        })
        
        # Print a single processing message
        print(f"🔄 Processing pair: {audio_file} and {robot_file}")
        
        # Construct the output file path
        output_filename = f"merged_{curvature_str}{test_id}.csv"
        merged_output_path = os.path.join(merged_dir, output_filename)
        
        # Load the audio and robot CSV files into pandas DataFrames
        print(f"📥 Loading CSV files...")
        df_audio = pd.read_csv(audio_path)  # Load the audio data
        df_robot = pd.read_csv(robot_path)  # Load the robot data
        
        # Check for required columns in robot data
        required_cols = ["Timestamp", "Clamp", "Section"]
        if not all(col in df_robot.columns for col in required_cols):
            print(f"❌ Robot CSV file missing required columns: {required_cols}")
            print(f"   Found columns: {df_robot.columns.tolist()}")
            print(f"   Skipping file: {robot_file}")
            continue
        
        # Add near where df_audio and df_robot are loaded
        if "Timestamp" not in df_audio.columns or "Timestamp" not in df_robot.columns:
            print(f"❌ ERROR: Missing Timestamp column in one of the files")
            print(f"   Audio columns: {df_audio.columns.tolist()[:5]}...")
            print(f"   Robot columns: {df_robot.columns.tolist()[:5]}...")
            continue
        
        print(f"🧾 Audio rows loaded: {len(df_audio)}")
        print(f"🤖 Robot rows loaded: {len(df_robot)}")

        # Parse the 'Timestamp' column in both DataFrames into datetime objects
        df_audio["Timestamp"] = pd.to_datetime(df_audio["Timestamp"]).dt.tz_localize(None)
        df_robot["Timestamp"] = pd.to_datetime(df_robot["Timestamp"]).dt.tz_localize(None)
        
        # Calculate time difference between robot and audio timestamps
        robot_time = df_robot['Timestamp'].min()
        audio_time = df_audio['Timestamp'].min()
        time_diff = robot_time - audio_time
        hours_diff = time_diff.total_seconds() / 3600
        
        # Adjust for timezone difference if close to 1 hour
        if 0.9 <= hours_diff <= 1.1:  # Approximately 1 hour difference
            print(f"🕒 Detected timezone difference of {hours_diff:.2f} hours between robot and audio timestamps")
            print(f"   Adjusting robot timestamps to match audio timezone...")
            
            # Keep original timestamps for reference
            df_robot["Original_Timestamp"] = df_robot["Timestamp"].copy()
            
            # Adjust robot timestamps by subtracting 1 hour
            df_robot["Timestamp"] = df_robot["Timestamp"] - pd.Timedelta(hours=1)
            
            print(f"   Updated robot timestamp range: {df_robot['Timestamp'].min()} to {df_robot['Timestamp'].max()}")
        
        # Sort the DataFrames by the 'Timestamp' column to ensure proper ordering
        df_audio.sort_values("Timestamp", inplace=True)
        df_robot.sort_values("Timestamp", inplace=True)
        
        # The 1-hour difference between timestamps exists because:
        # 1. Audio timestamps are recorded in UTC (from pd.Timestamp.now(tz='UTC'))
        # 2. Robot timestamps are recorded in local timezone (from datetime.datetime.now())
        # We handle this by adjusting robot timestamps when a ~1 hour difference is detected
        
        print(f"🕒 Audio timestamp range: {df_audio['Timestamp'].min()} to {df_audio['Timestamp'].max()}")
        print(f"🕒 Robot timestamp range: {df_robot['Timestamp'].min()} to {df_robot['Timestamp'].max()}")
        
        # Sanity check: robot should never start before audio
        robot_start_time = df_robot['Timestamp'].min()
        audio_start_time = df_audio['Timestamp'].min()

        if robot_start_time < audio_start_time:
            reason = (
                f"Robot started before audio recording.\n"
                f"   Robot start: {robot_start_time}\n"
                f"   Audio start: {audio_start_time}"
            )
            print(f"❌ ERROR: {reason}")
            print("   Skipping and moving these files to bad_data folder.\n")

            # Define bad_data folder path
            bad_data_dir = os.path.join(raw_dir, "bad_data")
            os.makedirs(bad_data_dir, exist_ok=True)

            # Move files
            shutil.move(robot_path, os.path.join(bad_data_dir, os.path.basename(robot_path)))
            shutil.move(audio_path, os.path.join(bad_data_dir, os.path.basename(audio_path)))

            # Add reason to the global list
            bad_file_reasons.append({
                "robot_file": os.path.basename(robot_path),
                "audio_file": os.path.basename(audio_path),
                "reason": reason
            })

            print(f"✓ Moved problematic files to {bad_data_dir}")
            print(f"  - {os.path.basename(robot_path)}")
            print(f"  - {os.path.basename(audio_path)}")
            return
        
        # Create a copy of audio data as the base for our merged dataframe
        merged_df = df_audio.copy()
        
        # Add/overwrite Curvature column with the curvature value from the filename
        # This will be the default curvature for active segments.
        merged_df["Curvature"] = curvature_value
        
        # Add the new Curvature_Active column (default to 0 - inactive)
        merged_df["Curvature_Active"] = 0
        
        # Add Position_cm column (initialized with NaN)
        merged_df["Position_cm"] = float('nan')
        
        # Add Section column if it doesn't exist (we'll remove it later)
        if "Section" not in merged_df.columns:
            merged_df["Section"] = ""
            
        print(f"🔍 Finding curvature activation intervals...")
        
        # Find all "close" events in the robot data
        close_events = df_robot[df_robot["Clamp"] == "close"].copy()
        print(f"   Found {len(close_events)} 'close' events")
        
        # If there are no close events, check case sensitivity
        if len(close_events) == 0:
            # Try other possible formats of "close" in the data
            for possible_close in ["Close", "CLOSE", "closed", "Closed", "CLOSED"]:
                test_events = df_robot[df_robot["Clamp"] == possible_close]
                if len(test_events) > 0:
                    print(f"   Found {len(test_events)} events with Clamp = '{possible_close}' instead")
                    # Use this value instead
                    close_events = test_events.copy()
                    # Also update the open event search below to use matching case
                    open_match = {"Close": "Open", "CLOSE": "OPEN", "closed": "opened", 
                                 "Closed": "Opened", "CLOSED": "OPENED"}
                    open_value = open_match.get(possible_close, "open")
                    break
        
        # Track intervals and matches for reporting
        interval_count = 0
        total_matches = 0
        
        # Process each "close" event to find corresponding "open" event
        for _, close_row in close_events.iterrows():
            close_time = close_row["Timestamp"]  # Now using adjusted timestamp
            section = close_row["Section"]
            
            # Find the next "open" event after this close event using full timestamps
            open_events = df_robot[(df_robot["Clamp"] == "open") & 
                                   (df_robot["Timestamp"] > close_time)]
            
            if open_events.empty:
                print(f"⚠️ Warning: No matching 'open' event found for 'close' at {close_time}")
                continue
                
            # Get the timestamp of the next open event
            open_time = open_events.iloc[0]["Timestamp"]
            
            # Find all audio rows that fall within this interval using full timestamps
            mask = (merged_df["Timestamp"] >= close_time) & (merged_df["Timestamp"] <= open_time)
            matches = mask.sum()
            
            if matches > 0:
                # Extract numeric value from Section
                position_cm = 0.0
                if section and "cm" in str(section):
                    position_cm = float(str(section).replace("cm", "").strip())
                
                # Mark these rows as having active curvature and set position
                merged_df.loc[mask, "Curvature_Active"] = 1
                merged_df.loc[mask, "Section"] = section
                merged_df.loc[mask, "Position_cm"] = position_cm
                
                interval_count += 1
                total_matches += matches
                
                print(f"✅ Interval {interval_count}: {close_time} to {open_time} - Position: {position_cm}cm")
                print(f"   Matched {matches} audio rows in this interval")
        
        # Print summary statistics
        print(f"\n📊 Interval matching statistics:")
        print(f"   - Found {interval_count} valid close-open intervals")
        print(f"   - Marked {total_matches} rows as Curvature_Active=1")
        print(f"   - Unmarked rows (Curvature_Active=0): {len(merged_df) - total_matches}")
        
        # Set Curvature to 0.0 for rows where Curvature_Active is 0.
        # For rows where Curvature_Active is 1, the original curvature_value is retained.
        # This aligns with the logic that if the sensor isn't actively measuring (no position propagated),
        # the curvature for that specific audio row should be considered 0.
        merged_df.loc[merged_df['Curvature_Active'] == 0, 'Curvature'] = 0.0
        print(f"✓ Updated 'Curvature' column: Set to 0.0 for {(merged_df['Curvature_Active'] == 0).sum()} inactive rows.")
        
        # Remove the original Section column
        if "Section" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Section"])
        
        # Save the merged DataFrame
        merged_df.to_csv(merged_output_path, index=False)
        
        # Add to merged summary
        merged_summary.append({
            "curvature": curvature_value,
            "test_id": test_id,
            "path": merged_output_path
        })
        
        # Print success message
        print(f"\n✅ Merge complete. Output saved to: {merged_output_path}")
        print(f"🧾 Rows in final CSV: {len(merged_df)}")
        
        # Add inside merge_csvs function
        print(f"🔍 First few rows of audio data:")
        print(f"   Columns: {df_audio.columns.tolist()[:5]}...")
        print(f"   Timestamp format: {df_audio['Timestamp'].iloc[0]}")

        # Add after timestamp processing
        print("\n🕰️ Sample timestamps for comparison:")
        if not df_robot.empty:
            print(f"   First robot timestamp: {df_robot['Timestamp'].iloc[0]}")
        if not df_audio.empty:
            print(f"   First audio timestamp: {df_audio['Timestamp'].iloc[0]}")
        print(f"   Are timestamps comparable: {type(df_robot['Timestamp'].iloc[0]) == type(df_audio['Timestamp'].iloc[0])}")

        '''
        # NOTE: This section below is commented out as it's a duplicate of code that already executed above.
        # The 1-hour difference between timestamps is because:
        # 1. Audio timestamps are recorded in UTC (from pd.Timestamp.now(tz='UTC'))
        # 2. Robot timestamps are recorded in local timezone (from datetime.datetime.now())
        # 
        # For systems in UTC+1 timezone (like Central European Time), this creates exactly 1 hour difference
        # even though recordings were done simultaneously.
        #
        # The minute:second matching approach addresses this by ignoring the hour part of timestamps
        # and comparing only minutes, seconds and milliseconds.

        # After loading timestamps, add this code to create minute:second versions for matching
        
        # Create minute-second normalized versions of timestamps for matching
        df_audio["MS_Time"] = df_audio["Timestamp"].apply(lambda x: x.strftime("%M:%S.%f"))
        df_robot["MS_Time"] = df_robot["Timestamp"].apply(lambda x: x.strftime("%M:%S.%f"))
        
        print(f"🕒 Example minute:second timestamps:")
        print(f"   Audio: {df_audio['MS_Time'].iloc[0]}")
        print(f"   Robot: {df_robot['MS_Time'].iloc[0]}")
        
        # Later, when finding intervals, use MS_Time instead of Timestamp
        for _, close_row in close_events.iterrows():
            close_time_ms = close_row["MS_Time"]
            section = close_row["Section"]
            close_time = close_row["Timestamp"]  # Keep for reporting
            
            # Find the next "open" event after this close event (by MS_Time)
            open_events = df_robot[(df_robot["Clamp"] == "open") & 
                                   (df_robot["MS_Time"] > close_time_ms)]
            
            if open_events.empty:
                print(f"⚠️ Warning: No matching 'open' event found for 'close' at {close_time}")
                continue
                
            # Get the timestamp of the next open event
            open_time = open_events.iloc[0]["Timestamp"]
            
            # Find all audio rows that fall within this interval
            mask = (merged_df["Timestamp"] >= close_time) & (merged_df["Timestamp"] <= open_time)
            matches = mask.sum()
            
            if matches > 0:
                # Extract numeric value from Section
                position_cm = 0.0
                if section and "cm" in str(section):
                    position_cm = float(str(section).replace("cm", "").strip())
                
                # Mark these rows as having active curvature and set position
                merged_df.loc[mask, "Curvature_Active"] = 1
                merged_df.loc[mask, "Section"] = section
                merged_df.loc[mask, "Position_cm"] = position_cm
                
                interval_count += 1
                total_matches += matches
                
                print(f"✅ Interval {interval_count}: {close_time} to {open_time} - Position: {position_cm}cm")
                print(f"   Matched {matches} audio rows in this interval")
        
        # Print summary statistics
        print(f"\n📊 Interval matching statistics:")
        print(f"   - Found {interval_count} valid close-open intervals")
        print(f"   - Marked {total_matches} rows as Curvature_Active=1")
        print(f"   - Unmarked rows (Curvature_Active=0): {len(merged_df) - total_matches}")
        
        # Set Curvature to 0.0 for rows where Curvature_Active is 0.
        # For rows where Curvature_Active is 1, the original curvature_value is retained.
        # This aligns with the logic that if the sensor isn't actively measuring (no position propagated),
        # the curvature for that specific audio row should be considered 0.
        merged_df.loc[merged_df['Curvature_Active'] == 0, 'Curvature'] = 0.0
        print(f"✓ Updated 'Curvature' column: Set to 0.0 for {(merged_df['Curvature_Active'] == 0).sum()} inactive rows.")
        
        # Remove the original Section column
        if "Section" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Section"])
        
        # Save the merged DataFrame
        merged_df.to_csv(merged_output_path, index=False)
        '''
def process_all_curvatures(raw_dir):
    """
    Processes all available curvature values found in the raw data directory.
    
    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw data files.
        
    Returns
    -------
    None
    """
    curvature_values = get_all_curvature_values(raw_dir)
    
    if not curvature_values:
        print("❌ No valid curvature values found in the raw data directory.")
        return
    
    print(f"🔍 Found {len(curvature_values)} unique curvature values: {curvature_values}")
    
    # Track all robot files to identify unpaired ones later
    robot_dir = os.path.join(raw_dir, "robot")
    all_robot_files = set([f for f in os.listdir(robot_dir) if f.endswith('.csv')])
    
    # First show all matched pairs before processing
    print("\n📋 Found file pairs to process:")
    for curvature in curvature_values:
        curvature_str = str(curvature).replace(".", "_")
        audio_pattern = f"raw_audio_{curvature_str}"
        audio_fft_dir = os.path.join(raw_dir, "audio_fft")
        
        audio_files = []
        for f in os.listdir(audio_fft_dir):
            if f.endswith(".csv") and f.startswith(audio_pattern):
                remaining = f[len(audio_pattern):]
                if not remaining or remaining.startswith("["):
                    audio_files.append(f)
        
        for audio_file in audio_files:
            test_id = ""
            if "[" in audio_file:
                test_id = audio_file[audio_file.index("["):audio_file.index("]")+1]
                
            robot_file = f"raw_robot_{curvature_str}{test_id}.csv"
            if os.path.exists(os.path.join(robot_dir, robot_file)):
                print(f"  Robot file: {robot_file}")
                print(f"  Audio file: {audio_file}\n")
    
    # Now process each curvature
    for curvature in curvature_values:
        print(f"\n⭐ Processing curvature value: {curvature}")
        merge_csvs(curvature)
    
    # Handle unpaired robot files after all merging is done
    used_robot_files = set([pair["robot_file"] for pair in processed_pairs])
    unused_robot_files = all_robot_files - used_robot_files
    
    # Create no_pairs directory if it doesn't exist
    no_pairs_dir = os.path.join(raw_dir, "no_pairs")
    os.makedirs(no_pairs_dir, exist_ok=True)
    
    # Move unpaired robot files to no_pairs directory
    for robot_file in unused_robot_files:
        robot_path = os.path.join(robot_dir, robot_file)
        shutil.move(robot_path, os.path.join(no_pairs_dir, robot_file))
        
        # Track this unpaired file
        unpaired_files.append({
            "file": robot_file,
            "type": "robot",
            "reason": "No matching audio file"
        })
        
        print(f"✓ Moved unpaired robot file to no_pairs: {robot_file}")
    
    # Print summary of successfully merged files
    if merged_summary:
        print("\n📄 Summary of successfully merged files:")
        for entry in merged_summary:
            print(f"  - Curvature: {entry['curvature']}, Test: {entry['test_id']} → {entry['path']}")
    else:
        print("ℹ️ No files were successfully merged.")

def main():
    """
    Entry point for the CSV Merger script.
    Prompts the user to process all curvatures or a specific curvature value.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "..", "csv_data", "raw")
    
    choice = input("Enter curvature value (e.g., 0.01818), or 'all' to process all curvatures: ")
    
    if choice.lower() == 'all':
        process_all_curvatures(raw_dir)
    else:
        try:
            curvature = float(choice)
            merge_csvs(curvature)
        except ValueError:
            print("❌ Invalid input. Please enter a valid curvature value or 'all'.")

    # Print the summary of processed pairs
    if processed_pairs:
        print("\n📋 Processed pairs:")
        for pair in processed_pairs:
            print(f"  - {pair['audio_file']} and {pair['robot_file']}")
    else:
        print("\n❌ No pairs were processed.")
        
    # Print summary of unpaired files
    if unpaired_files:
        print("\n📋 Summary of unpaired files moved to no_pairs directory:")
        for entry in unpaired_files:
            print(f"  - {entry['type'].capitalize()} file: {entry['file']}")
            print(f"    Reason: {entry['reason']}")
    else:
        print("\n✅ No unpaired files were found.")

    # Print the reasons for bad files at the end
    if bad_file_reasons:
        print("\n📋 Summary of bad files moved to bad_data:")
        for entry in bad_file_reasons:
            print(f"  - Robot file: {entry['robot_file']}")
            print(f"    Audio file: {entry['audio_file']}")
            print(f"    Reason: {entry['reason']}\n")
    else:
        print("\n✅ No files were moved to bad_data.")

def parse_filename(filename):
    """
    Parse a raw data filename to extract file type, curvature value, and test ID.
    
    Parameters
    ----------
    filename : str
        The filename to parse (e.g., "raw_audio_0_01818[test 1].csv")
        
    Returns
    -------
    dict
        Dictionary containing parsed components: file_type, curvature_value, test_id
        Returns None if the filename doesn't match the expected pattern
    """
    # Regex pattern to extract components from filenames
    # This handles cases like:
    # - raw_audio_0_01818[test 1].csv
    # - raw_robot_0_01818.[test 2].csv (with extra period)
    pattern = r'raw_(audio|robot)_([0-9_]+)(?:[.\s]*\[(test\s*\d+)\])?\.csv$'
    
    match = re.match(pattern, filename)
    if not match:
        return None
        
    file_type, curvature_str, test_id = match.groups()
    
    # Convert curvature string to float
    try:
        curvature_value = float(curvature_str.replace('_', '.'))
    except ValueError:
        return None
        
    # Default to empty string if no test ID was found
    test_id = test_id or ""
    
    return {
        "file_type": file_type,
        "curvature_value": curvature_value,
        "curvature_str": curvature_str,
        "test_id": test_id,
        "test_id_bracket": f"[{test_id}]" if test_id else ""
    }

if __name__ == "__main__":
    main()
