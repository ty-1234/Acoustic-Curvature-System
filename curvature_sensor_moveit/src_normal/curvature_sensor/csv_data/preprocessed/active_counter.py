import csv

def count_curvature_active(csv_file_path):
    count_0 = 0
    count_1 = 0

    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            if 'Curvature_Active' not in reader.fieldnames:
                print("Error: 'Curvature_Active' column not found in the CSV file.")
                return

            for row in reader:
                value = row['Curvature_Active']
                if value == '0':
                    count_0 += 1
                elif value == '1':
                    count_1 += 1

        print(f"Count of 0's: {count_0}")
        print(f"Count of 1's: {count_1}")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
csv_file_path = 'curvature_sensor_moveit/src_normal/curvature_sensor/csv_data/preprocessed/preprocessed_training_dataset.csv'  # Replace with the actual path to your CSV file
count_curvature_active(csv_file_path)