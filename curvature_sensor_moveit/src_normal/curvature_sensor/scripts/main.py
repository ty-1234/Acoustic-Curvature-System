"""
Curvature Sensing System Controller.

This script serves as the main entry point for the Curvature Sensing System, providing a
menu-driven interface to access various functions of the system:
- Data collection and processing tools
- ROS integration for robotics control
- Analysis and visualization utilities

Author: Bipindra Rai
Date: 2025-04-28
"""

import os
import sys
import importlib
import importlib.util
import subprocess
import signal

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def main():
    """
    Main function for the Curvature Sensing Pipeline Control Hub.
    """
    print("\n🔊 Curvature Sensing System Control Hub")
    print("═════════════════════════════════════════")
    print("Please select a task to run:")
    print("1. Generate WAV (2000hz_wav_creator.py)")
    print("2. Collect Data (curvature_data_collector.py)")
    print("3. ROS Control (curvature_ros.py)")
    print("4. Sync Robot and Audio CSV Files (csv_merger.py)")
    print("5. Normalize Data (baseline_norm.py)")
    print("6. Preprocess Data (preprocess_training_features.py)")
    print("═════════════════════════════════════════")

    try:
        choice = int(input("Enter the number corresponding to your choice: "))
    except ValueError:
        print("❌ Invalid input. Please enter a number between 1 and 6.")
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
            print("🔊 Starting frequency generator in background...")
            # Get the absolute path to frequency_gen.py
            freq_gen_path = os.path.join(os.path.dirname(__file__), "frequency_gen.py")
            
            # Start frequency_gen.py as a background process
            freq_process = subprocess.Popen([sys.executable, freq_gen_path])
            
            print("🔄 Starting data collection...")
            try:
                # Dynamically import the curvature_data_collector module and execute its main function
                curvature_data_collector = importlib.import_module("curvature_data_collector")
                curvature_data_collector.main()
                print("✅ Data collection completed successfully")
                print("──────────────────────────────────────────")
            finally:
                # Always terminate the frequency generator when data collection is done or if it fails
                print("🛑 Stopping frequency generator...")
                freq_process.terminate()
                freq_process.wait()  # Wait for process to terminate
                print("✅ Frequency generator stopped")
                
        except ImportError as e:
            # Handle missing module errors
            print(f"❌ Error: {e}. Ensure required scripts exist in the 'scripts' directory.")
        except Exception as e:
            print(f"❌ Error during data collection process: {e}")

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
            # Dynamically import the csv_merger module and execute its main function
            import csv_merger
            csv_merger.main()  # Call the main function of csv_merger.py
        except ImportError as e:
            print(f"❌ Error: {e}. Ensure 'csv_merger.py' exists in the 'scripts' directory.")
        except Exception as e:
            print(f"❌ Error while running csv_merger: {e}")

    # Task 5: Normalize data
    elif choice == 5:
        try:
            # Dynamically import the baseline_norm module and execute its main function
            baseline_norm = importlib.import_module("baseline_norm")
            print("🔄 Normalizing FFT columns...")
            baseline_norm.main()
            print("✅ Normalization completed successfully!")
        except ImportError as e:
            print(f"❌ Error: {e}. Ensure 'baseline_norm.py' exists in the 'scripts' directory.")
        except Exception as e:
            print(f"❌ Error during normalization: {e}")

    # Task 6: Preprocess data
    elif choice == 6:
        try:
            # Dynamically import the preprocess_training_features module
            module_path = os.path.join(os.path.dirname(__file__), "preprocess_training_features")
            spec = importlib.util.spec_from_file_location("preprocess_training_features", f"{module_path}.py")
            preprocess_training_features = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess_training_features)

            print("🔄 Processing training features...")
            preprocess_training_features.process_curvature_data()
            print("✅ Feature preprocessing completed successfully!")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Ensure 'preprocess_training_features.py' exists in the 'scripts' directory.")
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")

    # Handle invalid menu choices
    else:
        print("❌ Invalid choice. Please enter a number between 1 and 6.")

# Entry point of the script
if __name__ == "__main__":
    main()