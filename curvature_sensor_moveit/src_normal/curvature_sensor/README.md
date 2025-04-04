# Generate test signal
python main.py --task generate_wav

# Collect FFT data from sensor
python main.py --task collect_data

# Normalize and save processed CSV
python main.py --task preprocess

# Train ML model (soon)
python main.py --task train

# Visualize curvature estimates (soon)
python main.py --task visualize





