"""
Real-time prediction logger for curvature sensor.

This module logs real-time predictions of curvature and position values
to a CSV file with timestamps for later analysis and visualization.

Author: Bipindra Rai
Date: 2025-04-28
"""

import os
import csv
from datetime import datetime
import logging
import threading
import time

# Set up logging
logger = logging.getLogger(__name__)

class PredictionLogger:
    """
    Logger for real-time curvature and position predictions.
    
    This class handles logging prediction data to CSV files with timestamps
    and provides options for controlling logging frequency.
    """
    
    def __init__(self, output_dir=None, filename="predictions_log.csv", 
                 log_interval=1.0, create_new=False):
        """
        Initialize the prediction logger.
        
        Parameters:
        -----------
        output_dir : str or None
            Directory to save the log file (creates 'logs' dir if None)
        filename : str
            Name of the CSV log file
        log_interval : float
            Minimum time between logs in seconds
        create_new : bool
            If True, create a new file; if False, append to existing file
        """
        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(__file__), "logs")
        else:
            self.output_dir = output_dir
            
        # Create the directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure log file path
        self.log_path = os.path.join(self.output_dir, filename)
        
        # Logging parameters
        self.log_interval = log_interval
        self.last_logged_time = 0
        
        # Thread safety
        self.file_lock = threading.Lock()
        
        # Initialize log file
        self._initialize_log_file(create_new)
        
        logger.info(f"Prediction logger initialized. Log file: {self.log_path}")
        
    def _initialize_log_file(self, create_new):
        """
        Initialize the log file with headers if needed.
        
        Parameters:
        -----------
        create_new : bool
            If True, create a new file even if one exists
        """
        # Check if file exists
        file_exists = os.path.isfile(self.log_path)
        
        # Determine file mode (write or append)
        if create_new or not file_exists:
            mode = 'w'  # Create new file
            logger.info(f"Creating new log file: {self.log_path}")
        else:
            mode = 'a'  # Append to existing file
            logger.info(f"Appending to existing log file: {self.log_path}")
        
        # Initialize file with headers if creating new file
        with self.file_lock:
            with open(self.log_path, mode=mode, newline='') as file:
                writer = csv.writer(file)
                if mode == 'w':  # Only write headers for new files
                    # Updated to show mm^-1 instead of m^-1
                    writer.writerow(["Timestamp", "Curvature (mm^-1)", "Position (cm)"])
    
    def log_prediction(self, curvature, position, force=False):
        """
        Log a prediction to the CSV file with timestamp.
        
        Parameters:
        -----------
        curvature : float
            Predicted curvature value in mm^-1
        position : float
            Predicted position value in cm
        force : bool
            If True, log regardless of interval
            
        Returns:
        --------
        bool
            True if logged, False if skipped due to interval
        """
        # Get current time
        now = time.time()
        
        # Check if enough time has passed since last log
        time_since_last = now - self.last_logged_time
        
        if force or time_since_last >= self.log_interval:
            # Create timestamp string
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write to CSV file with thread safety
            with self.file_lock:
                try:
                    with open(self.log_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, f"{curvature:.6f}", f"{position:.2f}"])
                    
                    # Update last logged time
                    self.last_logged_time = now
                    
                    # Updated to show mm^-1 instead of m^-1
                    logger.info(f"Logged prediction - Curvature: {curvature:.6f} mm^-1, Position: {position:.2f} cm")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error logging prediction: {e}")
                    return False
        else:
            # Skip logging due to interval
            logger.debug(f"Skipping log (interval not reached): {time_since_last:.2f}s < {self.log_interval:.2f}s")
            return False
    
    def close(self):
        """Close the logger and perform any cleanup."""
        logger.info(f"Closing prediction logger. Log file: {self.log_path}")


# Global logger instance for module-level access
_logger = None

def init_logger(output_dir=None, filename="predictions_log.csv", 
                log_interval=1.0, create_new=False):
    """
    Initialize the global logger instance.
    
    Parameters:
    -----------
    output_dir : str or None
        Directory to save log files
    filename : str
        Name of the CSV log file
    log_interval : float
        Minimum time between logs in seconds
    create_new : bool
        If True, create a new file; if False, append to existing file
        
    Returns:
    --------
    PredictionLogger
        The logger instance
    """
    global _logger
    _logger = PredictionLogger(output_dir, filename, log_interval, create_new)
    return _logger

def log_data(curvature, position, force=False):
    """
    Log prediction data using the global logger instance.
    
    Parameters:
    -----------
    curvature : float
        Predicted curvature value
    position : float
        Predicted position value
    force : bool
        If True, log regardless of interval
        
    Returns:
    --------
    bool
        True if logged, False if skipped due to interval
    """
    global _logger
    
    # Initialize logger if not already done
    if _logger is None:
        _logger = init_logger()
    
    return _logger.log_prediction(curvature, position, force)

def close_logger():
    """Close the global logger instance."""
    global _logger
    if _logger is not None:
        _logger.close()
        _logger = None


# Test function to demonstrate logging
if __name__ == "__main__":
    # Configure logging for test
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create a test directory for logs
    test_dir = os.path.join(os.path.dirname(__file__), "test_logs")
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize logger with test settings
    init_logger(
        output_dir=test_dir,
        filename="test_predictions.csv",
        log_interval=0.5,  # Log every half second
        create_new=True    # Create a new file
    )
    
    # Log some test data
    print("Logging test predictions...")
    
    try:
        # Log 10 sample predictions
        for i in range(10):
            # Generate some test values
            curvature = 5.0 + (i * 0.1)   # 5.0, 5.1, 5.2, ...
            position = 2.0 + (i * 0.2)    # 2.0, 2.2, 2.4, ...
            
            # Log the prediction
            log_data(curvature, position)
            
            # Sleep to simulate real-time operation
            time.sleep(0.25)  # This will test the interval logic
    
    finally:
        # Ensure logger is closed properly
        close_logger()
        
    print(f"Test complete. Log file created at: {os.path.join(test_dir, 'test_predictions.csv')}")