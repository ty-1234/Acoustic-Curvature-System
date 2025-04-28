"""
Real-time prediction module for curvature sensor.

This module receives scaled FFT features and uses a pre-loaded model
to predict curvature and position values in real-time.

Author: Bipindra Rai
Date: 2025-04-28
"""

import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

def predict(model, features):
    """
    Make predictions using the pre-loaded model.
    
    Parameters:
    -----------
    model : sklearn model
        Pre-loaded prediction model (managed by orchestrator.py)
    features : ndarray
        Properly scaled feature array
        
    Returns:
    --------
    tuple
        (curvature, position) prediction values
    """
    try:
        # Ensure features are in the correct shape (2D array)
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Make prediction
        predictions = model.predict(features)
        
        # Extract curvature and position values from prediction
        # Assuming the model returns [position, curvature]
        if predictions.ndim > 1:
            position, curvature = predictions[0]
        else:
            position, curvature = predictions
            
        logger.debug(f"Prediction made: position={position:.2f}, curvature={curvature:.4f}")
        
        return position, curvature
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Return default values in case of error
        return -1.0, 0.0
        
def process_prediction(position, curvature):
    """
    Process raw prediction values to apply business logic.
    
    Parameters:
    -----------
    position : float
        Raw position prediction
    curvature : float
        Raw curvature prediction
        
    Returns:
    --------
    tuple
        (processed_position, processed_curvature, is_active)
    """
    # Determine if the curvature sensor is active based on position
    is_active = position > 0
    
    # Apply any additional processing to the prediction values
    # For example, enforce physical limits or smoothing
    processed_position = max(0, position) if is_active else -1
    processed_curvature = max(0, curvature) if is_active else 0
    
    return processed_position, processed_curvature, is_active

def validate_features(features):
    """
    Validate that the feature array contains valid data.
    
    Parameters:
    -----------
    features : ndarray
        Feature array to validate
        
    Returns:
    --------
    bool
        True if features are valid, False otherwise
    """
    if features is None:
        return False
        
    # Check for NaN or infinite values
    if np.isnan(features).any() or np.isinf(features).any():
        logger.warning("Features contain NaN or infinite values")
        return False
        
    # Check feature dimensionality
    if features.size == 0:
        logger.warning("Empty feature array")
        return False
        
    return True