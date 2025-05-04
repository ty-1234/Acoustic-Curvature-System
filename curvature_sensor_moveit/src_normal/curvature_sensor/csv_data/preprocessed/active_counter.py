import csv
import os
import glob

def count_curvature_active(directory_path):
    total_count_0 = 0
    total_count_1 = 0
    processed_files = 0
    error_files = 0
    
    # Get all preprocessed CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "preprocessed_*.csv"))
    
    if not csv_files:
        print(f"No preprocessed CSV files found in {directory_path}")
        return
    
    for csv_file_path in csv_files:
        file_count_0 = 0
        file_count_1 = 0
        file_name = os.path.basename(csv_file_path)
        
        try:
            with open(csv_file_path, mode='r') as file:
                reader = csv.DictReader(file)
                if 'Curvature_Active' not in reader.fieldnames:
                    print(f"Error: 'Curvature_Active' column not found in {file_name}")
                    error_files += 1
                    continue

                for row in reader:
                    value = row['Curvature_Active']
                    if value == '0':
                        file_count_0 += 1
                        total_count_0 += 1
                    elif value == '1':
                        file_count_1 += 1
                        total_count_1 += 1
            
            processed_files += 1
            print(f"File: {file_name}")
            print(f"  Count of 0's: {file_count_0}")
            print(f"  Count of 1's: {file_count_1}")
            print(f"  Total rows: {file_count_0 + file_count_1}")
            print("-----------------------------------")

        except FileNotFoundError:
            print(f"Error: File not found at {csv_file_path}")
            error_files += 1
        except Exception as e:
            print(f"An error occurred with {file_name}: {e}")
            error_files += 1

    print("\n=== SUMMARY ===")
    print(f"Total files processed: {processed_files}")
    print(f"Files with errors: {error_files}")
    print(f"Total count of 0's across all files: {total_count_0}")
    print(f"Total count of 1's across all files: {total_count_1}")
    print(f"Total rows processed: {total_count_0 + total_count_1}")
    if (total_count_0 + total_count_1) > 0:
        print(f"Percentage of 1's: {(total_count_1 / (total_count_0 + total_count_1)) * 100:.2f}%")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
count_curvature_active(current_dir)