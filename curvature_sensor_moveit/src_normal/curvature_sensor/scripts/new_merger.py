import pandas as pd
import glob
import os
from pathlib import Path

def load_and_prepare_csv(file_path):
    df = pd.read_csv(file_path)
    # Convert timestamp to datetime, handling both formats
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.replace('T', ' '), errors='coerce')
    # Ensure the Timestamp column is timezone-naive
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    return df

def merge_files():
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parent / 'csv_data' / 'raw2'
    output_dir = script_dir.parent / 'csv_data' / 'merged2'
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_files = list(raw_dir.glob('robot/raw_robot*.csv'))

    for robot_file in robot_files:
        try:
            robot_name = robot_file.name
            test_info = robot_name.replace('raw_robot', '')
            audio_file = raw_dir / f'audio_fft/raw_audio{test_info}'

            if not audio_file.exists():
                print(f"No matching audio file found for {robot_name}")
                continue

            print(f"\nProcessing:")
            print(f"Robot: {robot_name}")
            print(f"Audio: {audio_file.name}")

            # Load and prepare the CSV files
            robot_df = load_and_prepare_csv(robot_file)
            audio_df = load_and_prepare_csv(audio_file)

            # Shift the timestamps of the audio files by one hour
            audio_df['Timestamp'] = audio_df['Timestamp'] + pd.Timedelta(hours=1)

            # Ensure Timestamp columns are in datetime format
            robot_df['Timestamp'] = pd.to_datetime(robot_df['Timestamp'], errors='coerce')
            audio_df['Timestamp'] = pd.to_datetime(audio_df['Timestamp'], errors='coerce')

            # Sort by Timestamp to prepare for merge_asof
            robot_df = robot_df.sort_values('Timestamp')
            audio_df = audio_df.sort_values('Timestamp')

            # Perform the merge in the other direction
            merged_df = pd.merge_asof(
                audio_df,
                robot_df,
                on='Timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1s')
            )
            # Save the merged file
            output_file = output_dir / f'merged{test_info}'
            merged_df.to_csv(output_file, index=False)
            print(f"Saved merged file: {output_file.name}")

        except Exception as e:
            print(f"Error processing {robot_name}: {str(e)}")

if __name__ == "__main__":
    merge_files()