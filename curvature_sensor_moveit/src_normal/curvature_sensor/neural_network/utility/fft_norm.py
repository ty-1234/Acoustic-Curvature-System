"""
FFT_Norm.py - Utility for FFT Feature Normalization

This file provides a function to apply baseline normalization to FFT (Fast Fourier Transform)
features in a pandas DataFrame. The baseline is calculated from 'idle' periods
in the data and subtracted from 'active' periods.
"""

import pandas as pd
import numpy as np

def apply_fft_baseline_normalization(df_input, fft_columns, position_column, activity_column, verbose=False):
    """
    Applies FFT baseline normalization to the active parts of the input DataFrame.

    The baseline is calculated from rows where 'activity_column' is 0 and that appear
    before the first occurrence of 'position_column' being 0.0.
    This baseline is then subtracted from the 'fft_columns' of rows where
    'activity_column' is 1.

    Args:
        df_input (pd.DataFrame): The input DataFrame. It is not modified in place.
        fft_columns (list): List of strings, names of the FFT columns.
        position_column (str): Name of the position column.
        activity_column (str): Name of the activity/curvature_active column.
        verbose (bool): If True, print diagnostic messages.

    Returns:
        tuple: (pd.DataFrame, np.ndarray or None)
            - The first element is a DataFrame containing only the 'active' rows,
              with FFT features normalized if a baseline was successfully calculated
              and applied. If no active rows, an empty DataFrame is returned.
              This is a copy of the active portion of the input DataFrame.
            - The second element is the calculated idle_fft_baseline (1D numpy array),
              or None if a baseline could not be determined or applied.
    """
    # Check for required columns
    required_cols = fft_columns + [position_column, activity_column]
    if not all(col in df_input.columns for col in required_cols):
        if verbose:
            print("    Error: Input DataFrame is missing one or more required columns for FFT normalization.")
        # Return empty active_df and None baseline if critical columns are missing
        return pd.DataFrame(columns=df_input.columns), None

    idle_fft_baseline = None
    # Find the index of the first occurrence of position_column == 0.0
    first_pos_zero_indices = df_input[df_input[position_column] == 0.0].index
    
    if not first_pos_zero_indices.empty:
        first_pos_zero_idx = first_pos_zero_indices[0]
        # Select rows before the first position_column == 0.0 and where activity is 0
        idle_candidate_rows = df_input[
            (df_input.index < first_pos_zero_idx) & 
            (df_input[activity_column] == 0)
        ]
        if not idle_candidate_rows.empty:
            # Check for NaNs in FFT columns before calculating mean
            if not idle_candidate_rows[fft_columns].isnull().any().any():
                idle_fft_baseline = idle_candidate_rows[fft_columns].mean().values
                if verbose:
                    print(f"    Calculated idle FFT baseline from {len(idle_candidate_rows)} rows.")
            else:
                if verbose:
                    print(f"    Warning: NaNs found in FFT columns of idle_candidate_rows. Baseline not calculated.")
        else:
            if verbose:
                print(f"    Warning: No rows found with '{activity_column} == 0' before the first '{position_column} == 0.0'. Baseline not calculated.")
    else:
        if verbose:
            print(f"    Warning: No rows found with '{position_column} == 0.0'. Baseline not calculated.")
            
    # Process active data
    # Create a copy to avoid modifying the original DataFrame slices directly
    active_df = df_input[df_input[activity_column] == 1].copy() 
    
    if active_df.empty:
        if verbose:
            print(f"    No active data (where '{activity_column} == 1') found in the DataFrame.")
        # Return empty active_df and the baseline (which might be None or calculated)
        return active_df, idle_fft_baseline

    if idle_fft_baseline is not None:
        # Ensure baseline is a 1D array of the correct length
        if len(idle_fft_baseline) == len(fft_columns):
            # Apply baseline subtraction
            active_df.loc[:, fft_columns] = active_df[fft_columns].values - idle_fft_baseline
            if verbose:
                print(f"    Normalized FFT features for {len(active_df)} active rows using baseline subtraction.")
        else:
            if verbose:
                print(f"    Error: Baseline length ({len(idle_fft_baseline)}) does not match number of FFT columns ({len(fft_columns)}). Normalization not applied to active data.")
            # In this case, we might consider the baseline as "not applied" effectively
            # For clarity, let's ensure we return the baseline but indicate it wasn't used for subtraction here.
            # The active_df remains unnormalized in this specific error case.
    else:
        if verbose:
            print(f"    FFT features for active rows not normalized as no valid baseline was calculated.")
            
    return active_df, idle_fft_baseline

# Example Usage (optional, can be commented out or removed):
# if __name__ == '__main__':
#     # This is a placeholder for example usage or tests
#     # You would typically call this function from your main model training scripts.
#     
#     # Sample data (replace with actual data loading if testing)
#     data = {
#         'FFT_200Hz': [10, 11, 100, 110, 120, 15, 16],
#         'FFT_400Hz': [20, 22, 200, 220, 240, 25, 26],
#         'Position_cm': [1, 2, 0.0, 5, 6, 0.0, 8],
#         'Curvature_Active': [0, 0, 0, 1, 1, 0, 1]
#     }
#     sample_df = pd.DataFrame(data)
#     
#     fft_cols = ['FFT_200Hz', 'FFT_400Hz']
#     pos_col = 'Position_cm'
#     act_col = 'Curvature_Active'
#     
#     print("Running example FFT normalization:")
#     normalized_active_df, baseline = apply_fft_baseline_normalization(sample_df, fft_cols, pos_col, act_col, verbose=True)
#     
#     if baseline is not None:
#         print("\nCalculated Baseline:")
#         for i, col_name in enumerate(fft_cols):
#             print(f"  {col_name}: {baseline[i]:.2f}")
#     else:
#         print("\nNo baseline was calculated.")
#         
#     print("\nNormalized Active DataFrame:")
#     if not normalized_active_df.empty:
#         print(normalized_active_df)
#     else:
#         print("Empty (no active data or error during processing).")
