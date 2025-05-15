import pandas as pd
import numpy as np

def convert_curvature_to_radius(curvature_data, zero_curvature_radius_val=np.inf):
    """
    Converts curvature values (in mm^-1) to radius values (in mm).

    Args:
        curvature_data (pd.Series or float or int): 
            A pandas Series containing curvature values, or a single curvature value.
            Assumed to be in units of mm^-1.
        zero_curvature_radius_val (float, optional): 
            The value to return for radius when curvature is zero. 
            Defaults to np.inf, representing an infinitely large radius (flat surface).
            Set to np.nan or another numeric value if preferred.

    Returns:
        pd.Series or float: 
            A pandas Series with radius values in mm, or a single radius value in mm.
            Returns values as specified by `zero_curvature_radius_val` where input curvature is 0.
    """
    if isinstance(curvature_data, pd.Series):
        # Handle pandas Series
        radius_series = curvature_data.copy()
        # Locations where curvature is zero
        zero_curvature_mask = (curvature_data == 0)
        # Locations where curvature is non-zero
        non_zero_curvature_mask = ~zero_curvature_mask
        
        # Calculate radius for non-zero curvatures
        radius_series.loc[non_zero_curvature_mask] = 1 / curvature_data[non_zero_curvature_mask]
        
        # Assign specified value for zero curvatures
        radius_series.loc[zero_curvature_mask] = zero_curvature_radius_val
        
        return radius_series
        
    elif isinstance(curvature_data, (int, float)):
        # Handle single numerical value
        if curvature_data == 0:
            return zero_curvature_radius_val
        else:
            return 1 / curvature_data
    else:
        raise TypeError("Input 'curvature_data' must be a pandas Series or a numeric value (int/float).")

# Example Usage (optional, can be commented out or removed):
# if __name__ == '__main__':
#     # Example with a pandas Series
#     curvatures_mm_inv = pd.Series([0.005, 0.01, 0, -0.002, 0.05])
#     print(f"Input Curvatures (mm^-1):\n{curvatures_mm_inv}")
#     
#     radii_mm = convert_curvature_to_radius(curvatures_mm_inv)
#     print(f"\nConverted Radii (mm) (0 curvature -> inf):\n{radii_mm}")
# 
#     radii_mm_nan = convert_curvature_to_radius(curvatures_mm_inv, zero_curvature_radius_val=np.nan)
#     print(f"\nConverted Radii (mm) (0 curvature -> NaN):\n{radii_mm_nan}")
# 
#     radii_mm_large_val = convert_curvature_to_radius(curvatures_mm_inv, zero_curvature_radius_val=999999) # Using a large finite number
#     print(f"\nConverted Radii (mm) (0 curvature -> 999999):\n{radii_mm_large_val}")
# 
#     # Example with a single value
#     single_curvature = 0.007142 # mm^-1
#     single_radius = convert_curvature_to_radius(single_curvature)
#     print(f"\nSingle curvature {single_curvature} mm^-1 -> Radius: {single_radius:.2f} mm")
# 
#     zero_curvature = 0.0
#     zero_radius = convert_curvature_to_radius(zero_curvature)
#     print(f"Single curvature {zero_curvature} mm^-1 -> Radius: {zero_radius}")
# 
#     try:
#         convert_curvature_to_radius("not a series or number")
#     except TypeError as e:
#         print(f"\nError caught as expected: {e}")