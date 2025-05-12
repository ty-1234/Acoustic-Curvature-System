import numpy as np
import cv2

# This file handles contour-related operations:
# 1. Finds the largest contour in an edge-detected image (Note: main.py has its own ROI contour logic)
# 2. Estimates the curvature of a contour.

def find_largest_contour(edges):
    """
    Find the largest contour in an edge-detected image.
    Note: The main script (main.py) uses its own logic for finding contours within an ROI.
          This function might be for general use or testing.

    Args:
        edges (numpy.ndarray): Edge-detected image.

    Returns:
        numpy.ndarray: The largest contour based on area.

    Raises:
        ValueError: If no contours are found in the image.
    """
    # Find all contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Raise an error if no contours are found
    if not contours:
        raise ValueError("No contours found in find_largest_contour function.")
    # Return the largest contour based on area
    return max(contours, key=cv2.contourArea)

def menger_curvature_for_points(p1, p2, p3):
    """
    Calculates the Menger curvature (1/R) for three 2D points.
    R is the radius of the circumcircle defined by p1, p2, p3.
    Returns 0.0 if points are collinear or any two points are coincident to avoid division by zero.
    """
    # Calculate side lengths of the triangle p1p2p3
    d12 = np.linalg.norm(p1 - p2)
    d23 = np.linalg.norm(p2 - p3)
    d31 = np.linalg.norm(p3 - p1)

    # Calculate the area of the triangle using the determinant/shoelace formula
    # Area = 0.5 * |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
    area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + 
                       p2[0]*(p3[1] - p1[1]) + 
                       p3[0]*(p1[1] - p2[1]))

    # If area is very small (points are collinear) or any side length is zero, curvature is 0
    # Use a small epsilon for area to handle floating point inaccuracies for near-collinear points
    epsilon = 1e-7 
    if area < epsilon or d12 < epsilon or d23 < epsilon or d31 < epsilon:
        return 0.0

    # Circumradius R = (d12 * d23 * d31) / (4 * Area)
    # Curvature k = 1/R = (4 * Area) / (d12 * d23 * d31)
    denominator = d12 * d23 * d31
    
    # This check should ideally be covered by area < epsilon or side length < epsilon
    if denominator < epsilon: 
        return 0.0
        
    curvature_val = (4 * area) / denominator
    return curvature_val


def estimate_curvature(contour, window=5):
    """
    Estimate the curvature (1/R in pixels^-1) of a contour using Menger curvature.

    Args:
        contour (numpy.ndarray): Contour points, expected shape (N, 1, 2).
        window (int): Number of points on either side of the current point (p2)
                      to select p1 and p3. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - points_for_curvature (list of tuple): List of (x, y) coordinates of the contour points (p2)
                                                   for which curvature was calculated.
            - curvatures (list of float): List of curvature values (1/R in pixels^-1)
                                          for the corresponding points.
    """
    curvatures = []
    points_for_curvature = [] 

    if len(contour) < 2 * window + 1:
        print(f"Warning: Contour length ({len(contour)}) is too short for window size ({window}). Returning empty curvature.")
        return points_for_curvature, curvatures

    for i in range(window, len(contour) - window):
        # Ensure points are 1D arrays of 2 elements (x,y)
        p1 = contour[i - window][0].astype(float) 
        p2 = contour[i][0].astype(float)          
        p3 = contour[i + window][0].astype(float) 

        curv = menger_curvature_for_points(p1, p2, p3)
        
        curvatures.append(curv)
        points_for_curvature.append(tuple(p2.astype(int))) # Store as tuple of ints

    return points_for_curvature, curvatures