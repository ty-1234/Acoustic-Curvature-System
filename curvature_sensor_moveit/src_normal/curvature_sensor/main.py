"""
Curvature Sensing System Controller.

This script serves as the main entry point for the Curvature Sensing System, providing a
menu-driven interface to access various functions of the system. It dynamically loads
modules as needed to optimize memory usage.

Author: Bipindra Rai
Date: 2025-04-17
"""

import os
import sys
import importlib

# Add the scripts directory to the Python path for module resolution
# This ensures that Python can dynamically import modules located in the "scripts" directory.
scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

def main():
    """
    Main function for the Curvature Sensing Pipeline Control Hub.
    
    Provides a menu-based interface for selecting and executing various tasks related to
    the curvature sensing system, including:
    - Wave file generation for acoustic excitation
    - Data collection from the acoustic sensor
    - Robot control via ROS
    - Data processing and merging
    - Visualization of results
    
    Returns
    -------
    None
        This function handles execution but does not return any values
    """

    # Display the menu options to the user
    print("Curvature Sensing Pipeline Control Hub")
    print("Please select a task to run:")
    print("1. Generate WAV (2000hz_wav_creator.py)")
    print("2. Collect Data (curvature_data_collector.py)")
    print("3. ROS Control (curvature_ros.py)")
    print("4. Merge CSV Files (csv_merger.py)")
    print("5. Preprocess Data (preprocess_training_features.py)")
    print("6. Visualize FFT Data (fft_visualizer.py)")

    try:
        # Prompt the user to select a task
        choice = int(input("Enter the number corresponding to your choice: "))
    except ValueError:
        # Handle invalid input (non-integer values)
        print("Invalid input. Please enter a number between 1 and 6.")
        return

    # Task 1: Generate WAV files
    if choice == 1:
        try:
            # Dynamically import the 2000hz_wav_creator module and execute its main function
            wav_creator = importlib.import_module("2000hz_wav_creator")
            wav_creator.main()
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure '2000hz_wav_creator.py' exists in the 'scripts' directory.")

    # Task 2: Collect data
    elif choice == 2:
        try:
            # Dynamically import the curvature_data_collector module and execute its main function
            curvature_data_collector = importlib.import_module("curvature_data_collector")
            curvature_data_collector.main()
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure 'curvature_data_collector.py' exists in the 'scripts' directory.")

    # Task 3: ROS Control
    elif choice == 3:
        try:
            # Dynamically import the curvature_ros module and execute its main function
            curvature_ros = importlib.import_module("curvature_ros")
            curvature_ros.main()
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure 'curvature_ros.py' exists in the 'scripts' directory.")

    # Task 4: Merge CSV files
    elif choice == 4:
        try:
            # Dynamically import the csv_merger module
            csv_merger = importlib.import_module("csv_merger")
            raw_dir = os.path.join(os.path.dirname(__file__), "csv_data", "raw")
            merged_dir = os.path.join(os.path.dirname(__file__), "csv_data", "merged")

            # Check if the raw directory exists
            if not os.path.exists(raw_dir):
                print(f"⚠️ Raw directory not found: {raw_dir}")
                return

            # List all raw audio CSV files in the raw directory
            audio_files = [f for f in os.listdir(raw_dir) if f.startswith("raw_audio_") and f.endswith(".csv")]

            # Check if there are any raw audio files to process
            if not audio_files:
                print("⚠️ No raw audio files found in 'csv_data/raw/'")
                return

            # Ensure the merged directory exists
            os.makedirs(merged_dir, exist_ok=True)

            # Get unique curvature values from audio files
            curvatures = set()
            for audio_file in audio_files:
                # Extract the curvature value from the filename
                parts = audio_file.replace("raw_audio_", "").split("[")[0]
                curvature_str = parts.replace(".csv", "")
                curvature_val = float(curvature_str.replace("_", "."))
                curvatures.add(curvature_val)

            # Process each unique curvature value
            for curvature_val in curvatures:
                print(f"🔄 Merging data for curvature: {curvature_val}")
                csv_merger.merge_csvs(curvature_value=curvature_val)
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure 'csv_merger.py' exists in the 'scripts' directory.")

    # Task 5: Preprocess data
    elif choice == 5:
        try:
            # Dynamically import the preprocess_training_features module
            preprocess_training_features = importlib.import_module("preprocess_training_features")
            print("🔄 Processing training features...")
            preprocess_training_features.process_curvature_data()
            print("✅ Feature preprocessing completed successfully!")
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure 'preprocess_training_features.py' exists in the 'scripts' directory.")
        except Exception as e:
            # Handle other errors during processing
            print(f"❌ Error during preprocessing: {e}")

    # Task 6: Visualize FFT Data
    elif choice == 6:
        try:
            # Dynamically import the fft_visualizer module
            fft_visualizer = importlib.import_module("fft_visualizer")
            fft_visualizer.main()  # Call the main function in fft_visualizer
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure 'fft_visualizer.py' exists in the 'scripts' directory.")

    # Handle invalid menu choices
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")

# RULE: Modules are imported dynamically only when required to optimize memory usage and maintain clarity. 
# This approach is particularly beneficial as the experiment involves both a ROS PC and a laptop, 
# eliminating the need to load all modules simultaneously.

# Entry point of the script
if __name__ == "__main__":
    main()
