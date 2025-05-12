import pandas as pd
import numpy as np
import os
import glob

def process_data_for_discrete_extraction(df_original: pd.DataFrame, round_dp: int = 4) -> tuple[pd.DataFrame, list]:
    """
    Processes sensor data for discrete FFT extraction:
    - Converts timestamps to 'time_seconds'.
    - Robustly identifies all distinct active blocks based on original 'Curvature_Active'.
    - For EACH active block:
        - Deletes ~1 second of rows immediately PRECEDING the block's start.
        - Deletes the last ~1 second of rows FROM WITHIN the block itself.
          (If block is <=1s, the entire active block part is deleted).
    - Sets 'Curvature' & 'Position_cm' for remaining parts of active blocks.
    - Recalculates 'Curvature_Active'.

    Args:
        df_original: The input DataFrame from a single run.
        round_dp: Number of decimal places for thresholding Curvature_Active.

    Returns:
        A tuple containing:
            - df: The DataFrame with rows deleted/truncated.
            - active_blocks_original_info: List of dicts detailing identified original active blocks
                                           (indices refer to pre-deletion/truncation state).
    """
    df = df_original.copy()

    if df.empty:
        print("  Input DataFrame is empty. Returning as is.")
        return df_original, []

    try:
        df['timestamp_dt'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"  Error converting 'Timestamp' column to datetime: {e}")
        return df_original, [] 

    df['time_seconds'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()
    
    if 'Curvature' not in df.columns: df['Curvature'] = 0.0
    else: df['Curvature'] = df['Curvature'].astype(float)
    if 'Position_cm' not in df.columns: df['Position_cm'] = np.nan
    else: df['Position_cm'] = df['Position_cm'].astype(float)
    if 'Curvature_Active' not in df.columns:
        print("  'Curvature_Active' column missing. Cannot identify blocks.")
        return df_original, []

    active_blocks_original_info = []
    # Robust block identification using changes in 'Curvature_Active'
    # Create a Series that's True at the start of a new block of Curvature_Active values
    is_block_start_point = (df['Curvature_Active'].diff() != 0)
    # Cumulative sum of these starts creates a unique ID for each block of consecutive values
    block_id_series = is_block_start_point.cumsum()
    
    # Iterate through each group where Curvature_Active is 1
    active_segments_grouped = df[df['Curvature_Active'] == 1].groupby(block_id_series[df['Curvature_Active'] == 1])
    
    for _, group_df in active_segments_grouped:
        if not group_df.empty:
            start_idx = group_df.index.min()
            end_idx = group_df.index.max()
            
            if start_idx in df_original.index and end_idx in df_original.index:
                active_blocks_original_info.append({
                    'start': start_idx,
                    'end': end_idx,
                    'original_curvature_at_start': df_original.loc[start_idx, 'Curvature'],
                    'original_position_at_start': df_original.loc[start_idx, 'Position_cm']
                })
            else:
                print(f"  Warning: Block indices {start_idx}-{end_idx} (from groupby) not in df_original.index. Skipping this identified segment.")

    print(f"Identified {len(active_blocks_original_info)} active blocks:")
    for i, block_config in enumerate(active_blocks_original_info):
        print(f"  Block {i+1}: Original indices {block_config['start']} to {block_config['end']}.")

    indices_to_delete = []

    for block_num, block_config in enumerate(active_blocks_original_info):
        block_start_idx = block_config['start']
        block_end_idx = block_config['end'] 

        if not (block_start_idx in df.index and block_end_idx in df.index and block_start_idx <= block_end_idx):
            print(f"  Skipping processing for original block {block_num+1} ({block_start_idx}-{block_end_idx}) as indices are not valid in current df state (should not happen if block ID is from df).")
            continue
        
        # --- Set Curvature and Position for the original active block extent FIRST ---
        # This ensures that if part of the block remains, it has the correct constant curvature.
        C_orig_at_block_start = block_config['original_curvature_at_start']
        # Ensure we are using .loc for safety with potentially non-contiguous original indices from df_original
        df.loc[block_start_idx : block_end_idx, 'Position_cm'] = df_original.loc[block_start_idx : block_end_idx, 'Position_cm'].values
        df.loc[block_start_idx : block_end_idx, 'Curvature'] = C_orig_at_block_start
        
        # --- 1. Deletion of ~1s window BEFORE the current active block's start ---
        if block_start_idx > df.index.min(): 
            time_at_block_actual_start = df.loc[block_start_idx, 'time_seconds']
            deletion_window_start_time_lookback = time_at_block_actual_start - 1.0
            
            candidate_indices_before_block = df[
                (df['time_seconds'] >= deletion_window_start_time_lookback) &
                (df['time_seconds'] < time_at_block_actual_start) & 
                (df.index < block_start_idx) 
            ].index.tolist()

            if candidate_indices_before_block:
                print(f"  Block {block_num+1}: Marking {len(candidate_indices_before_block)} rows for deletion BEFORE its start (Idx {block_start_idx}). Time window: [{deletion_window_start_time_lookback:.3f}s, {time_at_block_actual_start:.3f}s).")
                indices_to_delete.extend(candidate_indices_before_block)
        
        # --- 2. Deletion of the last ~1s FROM WITHIN the current active block ---
        if block_start_idx <= block_end_idx : 
            time_at_this_block_true_end = df.loc[block_end_idx, 'time_seconds']
            start_of_last_one_second_portion = time_at_this_block_true_end - 1.0

            rows_to_delete_from_this_block_end = df[
                (df.index >= block_start_idx) &  
                (df.index <= block_end_idx) &    
                (df['time_seconds'] > start_of_last_one_second_portion) & # Use > to avoid issues if block is exactly 1s and we want to keep none
                (df['time_seconds'] <= time_at_this_block_true_end)      
            ].index.tolist()
            
            # Your example: 3(1), 4(1), 5(1) --> 3(1), 4(1) [delete 5(1)]
            # If T5 - 1s = T4, then rows where time > T4 and <= T5 get deleted. This is row 5.
            # If block is T3, T4 (0.5s long). T_end = T4. T_end - 1s = T4-1s.
            # All rows (T3, T4) will have time_seconds > (T4-1s). So, all get deleted. This matches "if <=1s, delete all".

            if rows_to_delete_from_this_block_end:
                print(f"  Block {block_num+1} (Orig Idx {block_start_idx}-{block_end_idx}): Marking {len(rows_to_delete_from_this_block_end)} rows from ITS END for deletion. Time window for deletion: ({start_of_last_one_second_portion:.3f}s, {time_at_this_block_true_end:.3f}s].")
                indices_to_delete.extend(rows_to_delete_from_this_block_end)
                            
    if indices_to_delete:
        unique_indices_to_delete = sorted(list(set(idx for idx in indices_to_delete if idx in df.index)))
        if unique_indices_to_delete:
            print(f"  Attempting to delete {len(unique_indices_to_delete)} unique rows...")
            df.drop(index=unique_indices_to_delete, inplace=True)
            print(f"  Deletion complete. DataFrame shape now: {df.shape}")
        else:
            print("  No valid rows identified for deletion after checking current df indices.")
    else:
        print("  No rows identified for any deletion by the script logic.")
            
    if 'Curvature' in df.columns and not df.empty:
        threshold = 10**-(round_dp + 1) 
        df['Curvature_Active'] = np.where(df['Curvature'] > threshold, 1, 0)
        print(f"  'Curvature_Active' column updated (active if > {threshold}).")
    elif df.empty and 'Curvature_Active' in df.columns:
        df['Curvature_Active'] = pd.Series(dtype='int')
    elif 'Curvature' not in df.columns:
         print("  Warning: 'Curvature' column not found. Cannot update 'Curvature_Active'.")

    # Clean up helper columns that might have been added to df
    df.drop(columns=['timestamp_dt', 'time_seconds', 'active_diff'], inplace=True, errors='ignore')
    return df, active_blocks_original_info

# ==============================================================================
# The if __name__ == "__main__": block below should be YOUR existing main block
# from noise_remover.py or interpolator_feature.py.
# I am providing a generic placeholder here.
# Make sure to adjust input_directory, output_directory, and how you call
# process_data_for_discrete_extraction with your actual file looping.
# ==============================================================================
if __name__ == "__main__":
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output directories relative to the script's location
    input_directory = os.path.join(script_dir, "csv_data/merged/")
    output_directory = os.path.join(script_dir, "csv_data/cleaned/")
    # ==============================================================

    rounding_decimal_places_for_ca_threshold = 4 

    if "YOUR_INPUT_DIRECTORY_HERE" in input_directory or "YOUR_OUTPUT_DIRECTORY_HERE" in output_directory:
        # This check is a fallback if script_dir logic somehow fails or is removed.
        # For robust pathing, the os.path.join(script_dir, ...) is preferred.
        print("Error: Please ensure input_directory and output_directory are correctly set.")
        print(f"Current input_directory: {input_directory}")
        print(f"Current output_directory: {output_directory}")
        exit()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Use glob to process all relevant CSV files from your input directory
    csv_files_to_process = glob.glob(os.path.join(input_directory, "merged_*.csv")) 

    if not csv_files_to_process:
        print(f"ERROR: No CSV files found in '{input_directory}' matching the pattern 'merged_*.csv'. Please check the path and pattern.")
        exit()
    
    print(f"Found {len(csv_files_to_process)} file(s) to process in '{input_directory}'.")

    for csv_file_path in csv_files_to_process:
        base_name = os.path.basename(csv_file_path)
        print(f"\n--- Processing file: {base_name} ---")
        try:
            original_data = pd.read_csv(csv_file_path)
            if original_data.empty:
                print(f"  Loaded DataFrame from '{base_name}' is empty. Skipping.")
                continue
            print(f"  Successfully loaded data from '{base_name}'. Original Shape: {original_data.shape}")
        except Exception as e:
            print(f"  ERROR loading CSV '{base_name}': {e}. Skipping.")
            continue
        
        # Pass a copy to the processing function
        processed_df, _ = process_data_for_discrete_extraction(
            original_data.copy(), 
            round_dp=rounding_decimal_places_for_ca_threshold
        )
        print(f"  Processing finished for '{base_name}'. Shape after processing: {processed_df.shape}")

        # Construct output filename: replace "merged_" with "cleaned_"
        if base_name.startswith("merged_"):
            new_base_name = "cleaned_" + base_name[len("merged_"):]
        else:
            # Fallback if the file doesn't start with "merged_", prepend "cleaned_"
            new_base_name = "cleaned_" + base_name
            print(f"  Warning: Input filename '{base_name}' did not start with 'merged_'. Outputting as '{new_base_name}'.")
            
        output_file_path = os.path.join(output_directory, new_base_name)

        try:
            processed_df.to_csv(output_file_path, index=False) 
            print(f"  Successfully saved processed data to '{output_file_path}'")
        except Exception as e:
            print(f"  ERROR saving CSV to '{output_file_path}': {e}")
        
        print(f"--- Finished processing file: {base_name} ---")

    print("\nAll specified CSV files processed.")