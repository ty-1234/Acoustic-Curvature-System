import pandas as pd
import numpy as np
import os
import glob

def preprocess_curvature_rounded_ramp(df_original: pd.DataFrame, round_dp: int = 4) -> tuple[pd.DataFrame, list]:
    """
    Processes sensor data:
    - Converts timestamps.
    - Identifies active blocks based on original 'Curvature_Active'.
    - Generates leading curvature ramps (0 to peak), propagates start position.
    - Sets core active block curvature.
    - Implements NEW trailing curvature ramp logic:
        - Ramp down from original peak curvature.
        - The original block end index now holds the smallest positive step of this ramp.
        - Ramp occurs in ~1s ending at and including the original block end index.
        - Propagates original end position to this ramp.
    - Rounds ramped curvature values.

    Args:
        df_original: The input DataFrame from a single run.
        round_dp: Number of decimal places to round the ramp curvature values to.

    Returns:
        A tuple containing:
            - df: The DataFrame with 'Curvature' and 'Position_cm' columns modified.
            - active_blocks: A list of dictionaries detailing identified original active blocks.
    """
    df = df_original.copy()

    try:
        df['timestamp_dt'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"Error converting 'Timestamp' column to datetime: {e}")
        return df_original, [] 

    df['time_seconds'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()
    df['active_diff'] = df['Curvature_Active'].diff()

    block_start_indices = df[df['active_diff'] == 1].index.tolist()
    if not df.empty and df['Curvature_Active'].iloc[0] == 1:
        block_start_indices.insert(0, df.index[0])

    block_end_candidate_indices = df[df['active_diff'] == -1].index.tolist()
    actual_block_end_indices = []
    for end_idx_after_change in block_end_candidate_indices:
        if end_idx_after_change > 0:
            actual_block_end_indices.append(end_idx_after_change - 1)

    if not df.empty and df['Curvature_Active'].iloc[-1] == 1:
        last_idx = df.index[-1]
        if not actual_block_end_indices or actual_block_end_indices[-1] < last_idx:
            possible_starts_for_this_end = [s for s in block_start_indices if s <= last_idx]
            if possible_starts_for_this_end:
                last_potential_start = max(possible_starts_for_this_end)
                if not actual_block_end_indices or last_potential_start > (actual_block_end_indices[-1] if actual_block_end_indices else -1) :
                    actual_block_end_indices.append(last_idx)
    
    active_blocks = [] 
    last_end_processed = -1
    unique_sorted_starts = sorted(list(set(block_start_indices)))
    unique_sorted_ends = sorted(list(set(actual_block_end_indices)))

    for start_idx_loop in unique_sorted_starts:
        if start_idx_loop <= last_end_processed:
            continue
        corresponding_ends = [end_idx_loop for end_idx_loop in unique_sorted_ends if end_idx_loop >= start_idx_loop]
        if corresponding_ends:
            current_end_idx = corresponding_ends[0]
            if current_end_idx > last_end_processed:
                 active_blocks.append({
                     'start': start_idx_loop, 'end': current_end_idx, 
                     'original_curvature_at_start': df_original.loc[start_idx_loop, 'Curvature'],
                     'original_curvature_at_end_peak': df_original.loc[current_end_idx, 'Curvature'],
                     'original_position_at_start': df_original.loc[start_idx_loop, 'Position_cm'],
                     'original_position_at_end': df_original.loc[current_end_idx, 'Position_cm'] 
                 })
                 last_end_processed = current_end_idx
        elif not df.empty and df.loc[start_idx_loop, 'Curvature_Active'] == 1 and \
             (not unique_sorted_ends or start_idx_loop > max(unique_sorted_ends, default=-1)):
            current_end_idx = df.index[-1]
            active_blocks.append({
                'start': start_idx_loop, 'end': current_end_idx,
                'original_curvature_at_start': df_original.loc[start_idx_loop, 'Curvature'],
                'original_curvature_at_end_peak': df_original.loc[current_end_idx, 'Curvature'],
                'original_position_at_start': df_original.loc[start_idx_loop, 'Position_cm'],
                'original_position_at_end': df_original.loc[current_end_idx, 'Position_cm']
            })
            last_end_processed = current_end_idx
            break 

    print(f"Identified {len(active_blocks)} active blocks (based on original Curvature_Active):")
    for i, block_info in enumerate(active_blocks):
        start_val = block_info.get('start', -1)
        end_val = block_info.get('end', -1)
        if not df.empty and 0 <= start_val < len(df) and 0 <= end_val < len(df):
            orig_curv_peak_val = block_info.get('original_curvature_at_end_peak', float('nan'))
            # Using f-string directly for formatting round_dp
            print(f"  Block {i+1}: Original active indices {start_val} to {end_val}. Original Peak Curv at End: {orig_curv_peak_val:.{round_dp}f}")
        else: print(f"  Block {i+1}: Invalid block indices start={start_val}, end={end_val}")


    for block_num, block_config in enumerate(active_blocks):
        block_start_idx = block_config['start']
        block_end_idx = block_config['end'] 

        if not (not df.empty and 0 <= block_start_idx < len(df) and 0 <= block_end_idx < len(df) and block_start_idx <= block_end_idx):
            print(f"  Skipping invalid block {block_num+1}")
            continue
        
        C_orig_at_block_start = block_config['original_curvature_at_start']
        C_orig_at_block_end_peak = block_config['original_curvature_at_end_peak'] 
        Pos_at_block_start = block_config['original_position_at_start']
        Pos_at_block_end = block_config['original_position_at_end']
        
        df.loc[block_start_idx : block_end_idx, 'Position_cm'] = df_original.loc[block_start_idx : block_end_idx, 'Position_cm']
        if block_start_idx <= block_end_idx : 
             df.loc[df.index[block_start_idx : block_end_idx + 1], 'Curvature'] = C_orig_at_block_start 
        
        if block_start_idx > 0:
            time_at_block_actual_start = df.loc[block_start_idx, 'time_seconds']
            ramp_window_start_time = time_at_block_actual_start - 1.0
            candidate_indices = df[
                (df['time_seconds'] >= ramp_window_start_time) &
                (df['time_seconds'] < time_at_block_actual_start) &
                (df.index < block_start_idx)].index

            if not candidate_indices.empty:
                first_ramp_idx = candidate_indices.min()
                last_ramp_idx = block_start_idx - 1
                if first_ramp_idx <= last_ramp_idx:
                    num_ramp_rows = last_ramp_idx - first_ramp_idx + 1
                    if num_ramp_rows > 0:
                        total_intervals = num_ramp_rows + 1 
                        step_increment = C_orig_at_block_start / total_intervals
                        for j in range(num_ramp_rows):
                            current_idx_in_ramp = first_ramp_idx + j
                            raw_value = step_increment * (j + 1)
                            df.loc[current_idx_in_ramp, 'Curvature'] = round(max(0, raw_value), round_dp)
                            df.loc[current_idx_in_ramp, 'Position_cm'] = Pos_at_block_start
        
        if block_start_idx <= block_end_idx : 
            time_ramp_should_end_at_idx = df.loc[block_end_idx, 'time_seconds']
            time_ramp_should_start_target = time_ramp_should_end_at_idx - 1.0
            
            trail_ramp_candidate_indices = df[
                (df['time_seconds'] >= time_ramp_should_start_target) &
                (df['time_seconds'] <= time_ramp_should_end_at_idx) & 
                (df.index <= block_end_idx) 
            ].index

            if not trail_ramp_candidate_indices.empty:
                first_actual_idx_of_ramp = max(trail_ramp_candidate_indices.min(), block_start_idx)
                rows_to_receive_ramp_values = df.index[df.index.isin(range(first_actual_idx_of_ramp, block_end_idx + 1))]

                if not rows_to_receive_ramp_values.empty:
                    num_ramp_points = len(rows_to_receive_ramp_values)

                    if num_ramp_points > 0:
                        # print(f"  Block {block_num+1} (End: Idx {block_end_idx}): Applying FINAL trailing ramp ({num_ramp_points} rows).") # Reduced verbosity
                        if num_ramp_points == 1:
                            df.loc[block_end_idx, 'Curvature'] = round(C_orig_at_block_end_peak, round_dp)
                            df.loc[block_end_idx, 'Position_cm'] = Pos_at_block_end
                        else:
                            step_val_for_ramp = C_orig_at_block_end_peak / num_ramp_points
                            for j, current_idx in enumerate(rows_to_receive_ramp_values): 
                                raw_value = step_val_for_ramp * (num_ramp_points - j)
                                df.loc[current_idx, 'Curvature'] = round(max(0, raw_value), round_dp)
                                df.loc[current_idx, 'Position_cm'] = Pos_at_block_end
                            
    df.drop(columns=['timestamp_dt', 'time_seconds', 'active_diff'], inplace=True, errors='ignore')
    return df, active_blocks

# --- How to use the function (main part of your script) ---
if __name__ == "__main__":
    input_directory = "csv_data/merged/" 
    output_directory = "csv_data/interpolated/" # CHANGED: Define an output directory
    rounding_decimal_places = 4 

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    csv_files = glob.glob(os.path.join(input_directory, "merged_*.csv")) # Ensure we only process "merged_" files

    if not csv_files:
        exit(f"ERROR: No CSV files starting with 'merged_' found in '{input_directory}'. Please check the path.")

    all_processed_blocks_info = {} 

    for csv_file_path in csv_files:
        base_name = os.path.basename(csv_file_path)
        print(f"\n--- Processing file: {base_name} ---")
        try:
            original_data_for_script = pd.read_csv(csv_file_path)
            print(f"Successfully loaded data from '{base_name}'.")
        except Exception as e:
            print(f"ERROR loading CSV '{base_name}': {e}. Skipping.")
            continue

        if original_data_for_script.empty:
            print(f"ERROR: Loaded DataFrame from '{base_name}' is empty. Skipping.")
            continue

        original_curvature_snapshot_for_file = original_data_for_script['Curvature'].copy()

        print(f"Starting ramp processing for '{base_name}' (Curvature to {rounding_decimal_places} d.p.)...")
        processed_df, active_blocks_info_for_file = preprocess_curvature_rounded_ramp(original_data_for_script, round_dp=rounding_decimal_places)
        all_processed_blocks_info[base_name] = active_blocks_info_for_file
        print("Ramp processing finished.")

        if 'Curvature' in processed_df.columns and 'Curvature_Active' in processed_df.columns:
            print("Recalculating 'Curvature_Active'...")
            threshold = 10**-(rounding_decimal_places + 1) 
            processed_df['Curvature_Active'] = np.where(processed_df['Curvature'] > threshold, 1, 0)
            print(f"'Curvature_Active' column updated (active if Curvature > {threshold}).")
        else:
            missing_cols_ca = [col for col in ["'Curvature'", "'Curvature_Active'"] if col.strip("'") not in processed_df.columns]
            if missing_cols_ca: print(f"Warning: Could not update 'Curvature_Active'. Missing: {', '.join(missing_cols_ca)}.")

        print(f"\nEXAMPLE OUTPUT for {base_name}:")
        print("="*70)
        display_cols = ['Timestamp', 'Curvature_Active', 'Curvature', 'Position_cm']
        
        original_float_format = pd.options.display.float_format
        pd.options.display.float_format = f'{{:.{rounding_decimal_places+1}f}}'.format 

        print("\nFirst 5 rows of processed data:")
        print(processed_df[display_cols].head(5).to_string())
        print("\nLast 5 rows of processed data:")
        print(processed_df[display_cols].tail(5).to_string())
        
        if active_blocks_info_for_file and 'Curvature_Active' in processed_df.columns:
            points_to_show_briefly = []
            for block_cfg in active_blocks_info_for_file[:2]: 
                points_to_show_briefly.append(block_cfg['start'])
                points_to_show_briefly.append(block_cfg['end'])
            points_to_show_briefly = sorted(list(set(points_to_show_briefly)))
            points_to_show_briefly = [idx for idx in points_to_show_briefly if 0 <= idx < len(processed_df)]

            if points_to_show_briefly:
                 print(f"\nBrief look at data around original block boundaries:")
            for idx_of_interest in points_to_show_briefly:
                start_display = max(0, idx_of_interest - 3) 
                end_display = min(len(processed_df), idx_of_interest + 4)
                original_curv_display = "N/A"
                if idx_of_interest in original_curvature_snapshot_for_file.index:
                    original_curv_display = f"{original_curvature_snapshot_for_file.loc[idx_of_interest]:.{rounding_decimal_places}f}"
                
                print(f"\nAround index {idx_of_interest} (Original Curv for this Idx: {original_curv_display}):")
                # Temporarily set format for this specific print if needed, though global option should cover it
                # with pd.option_context('display.float_format', f'{{:.{rounding_decimal_places+1}f}}'.format):
                #    print(processed_df[display_cols].iloc[start_display:end_display].to_string())
                print(processed_df[display_cols].iloc[start_display:end_display].to_string())

        pd.options.display.float_format = original_float_format

        try:
            # --- CHANGED: Construct new output filename and path ---
            new_base_name = base_name # Fallback if not starting with "merged_"
            if base_name.startswith("merged_"):
                new_base_name = "interpolated_" + base_name[len("merged_"):]
            
            output_file_path = os.path.join(output_directory, new_base_name)
            # --- END CHANGE ---

            processed_df.to_csv(output_file_path, index=False, float_format=f'%.{rounding_decimal_places+2}f')
            print(f"\n(Successfully processed data saved to '{output_file_path}')")
        except Exception as e:
            print(f"ERROR saving CSV to '{output_file_path}': {e}")
        
        print(f"--- Finished processing file: {base_name} ---")

    print("\nAll CSV files in the directory processed.")
    pd.options.display.float_format = None