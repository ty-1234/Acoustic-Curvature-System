import argparse
import os
from scripts import (
    curvature_data_processor,   # Handles normalization of FFT data
    curvature_data_collector,  # Handles audio recording + FFT + raw CSV creation
)

# ========== TASKS ==========

# 🎵 Task: Generate test signal (200–2000 Hz tones)
def generate_wav():
    from scripts import wav_creator as wav
    wav.main()  # Your wav_creator.py should define a main() to run generation

# 🎙️ Task: Record FFT data for all sections using a known curvature object
def collect_data():
    from scripts import curvature_data_collector as collector
    collector.main()  # Your curvature_data_collector.py must have a main() method

# 🧹 Task: Preprocess all raw files in batch, normalize FFT features, save to preprocessed/
def preprocess_data():
    from scripts import curvature_data_processor as processor
    raw_dir = "csv_data/raw"

    # Get all files that start with raw_ and end with .csv
    files = [f for f in os.listdir(raw_dir) if f.startswith("raw_") and f.endswith(".csv")]

    if not files:
        print("⚠️ No raw files found in 'csv_data/raw/'")
        return

    for filename in files:
        # Extract curvature from filename: e.g. raw_0_01818.csv → 0.01818
        curvature_str = filename.replace("raw_", "").replace(".csv", "")
        curvature_val = float(curvature_str.replace("_", "."))

        print(f"🔄 Processing: {filename} (curvature: {curvature_val})")
        processor.process_curvature_data(curvature_value=curvature_val)

# ========= FUTURE TASKS =========

# 🔧 Task: Train your regression model (GPR or other)
def train_model():
    # TODO: Implement model training pipeline using preprocessed data
    print("🧠 Model training not implemented yet.")

# 📈 Task: Visualize model predictions vs ground truth
def visualize():
    # TODO: Implement plotting logic for model output and curvature profiles
    print("📊 Visualization not implemented yet.")

# ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curvature Sensor Control Panel")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["generate_wav", "collect_data", "preprocess", "train", "visualize"],
        help="Choose a task to run from the pipeline"
    )

    args = parser.parse_args()

    # Run the selected task
    if args.task == "generate_wav":
        generate_wav()
    elif args.task == "collect_data":
        collect_data()
    elif args.task == "preprocess":
        preprocess_data()
    elif args.task == "train":
        train_model()  # Placeholder for future model training
    elif args.task == "visualize":
        visualize()    # Placeholder for future visualization
