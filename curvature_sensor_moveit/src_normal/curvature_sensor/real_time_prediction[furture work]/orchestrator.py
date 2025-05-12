"""
Main entry point for real-time curvature prediction system.

This script integrates all components of the real-time prediction system:
- Frequency generation (audio output)
- Audio capture and feature extraction
- Model prediction
- Real-time visualization
- Logging of predictions

The orchestrator ensures proper sequencing and synchronization between components
to maintain accurate and reliable predictions.

Author: Bipindra Rai
Date: 2025-04-28
"""

import os
import time
import threading
import logging
import signal
import sys
import joblib
import sounddevice as sd  

# Import component modules
from rt_frequency_gen import RealTimeFrequencyGenerator
from rt_feature_extraction import RealTimeFeatureExtractor
from rt_model import predict, process_prediction, validate_features
from rt_logger import init_logger, close_logger
from rt_gui import PredictionGUI

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Central orchestrator for the real-time curvature prediction system.
    
    Coordinates all components and ensures proper timing between 
    frequency generation, audio capture, and prediction.
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the orchestrator.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the trained model file
        scaler_path : str or None
            Path to the feature scaler file
        """
        # Set default paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "model")
        
        self.model_path = model_path or os.path.join(model_dir, "extratrees_optuna_model.pkl")
        self.scaler_path = scaler_path or os.path.join(model_dir, "feature_scaler.pkl")
        
        # Components
        self.model = None
        self.scaler = None
        self.freq_generator = None
        self.feature_extractor = None
        self.gui = None
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Threads
        self.threads = {}
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Orchestrator initialized")
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals by stopping all components."""
        logger.info("Received termination signal")
        self.stop()
    
    def load_model_and_scaler(self):
        """Load the prediction model and feature scaler."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler file not found: {self.scaler_path}")
                return False
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            logger.info(f"Scaler expects {self.scaler.n_features_in_} features")
            if hasattr(self.scaler, 'feature_names_in_'):
                logger.info(f"Scaler feature names: {self.scaler.feature_names_in_}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            return False
    
    def initialize_components(self):
        """
        Initialize all system components.
        
        Returns:
        --------
        bool
            True if initialization succeeded, False otherwise
        """
        try:
            # Initialize frequency generator
            self.freq_generator = RealTimeFrequencyGenerator()
            
            # Initialize feature extractor
            self.feature_extractor = RealTimeFeatureExtractor()
            
            # Initialize GUI
            self.gui = PredictionGUI(update_interval=100)  # Update GUI every 100ms
            
            # Initialize logger
            log_filename = f"curvature_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            init_logger(filename=log_filename, log_interval=1.0, create_new=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def start(self):
        """Start all components of the system."""
        try:
            # Step 1: Load model and scaler
            if not self.load_model_and_scaler():
                logger.error("Failed to load model and scaler")
                return False
                
            # Step 2: Initialize components
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
                
            # Step 3: Start prediction thread
            logger.info("Starting prediction thread...")
            self.threads["prediction"] = threading.Thread(target=self.prediction_loop)
            self.threads["prediction"].daemon = True
            self.threads["prediction"].start()
            logger.info("Prediction loop started")
            
            # Step 4: Start frequency generator
            logger.info("Starting frequency generator...")
            if not self.freq_generator.start():
                logger.error("Failed to start frequency generator")
                return False
            
            # Step 5: Start feature extraction
            logger.info("Starting feature extraction...")
            if not self.feature_extractor.start():
                logger.error("Failed to start feature extraction")
                return False
                
            logger.info("Feature extraction thread started")
            
            # Get microphone information for the GUI
            try:
                devices = sd.query_devices()
                default_input = sd.default.device[0]
                device_info = devices[default_input]
                mic_name = device_info['name']
                
                # Update GUI with microphone info
                if self.gui:
                    self.gui.update_mic_info(mic_name)
            except Exception as e:
                logger.warning(f"Could not get microphone info for GUI: {e}")
                # Use a fallback
                if self.gui:
                    self.gui.update_mic_info("Default Microphone")
            
            # Set is_running flag to true after all components have started
            self.is_running = True
            
            logger.info("System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False
    
    def stop(self):
        """
        Stop the real-time prediction system.
        
        Returns:
        --------
        bool
            True if stopped successfully, False otherwise
        """
        if not self.is_running:
            return True
        
        logger.info("Stopping real-time prediction system...")
        
        # Set flags to stop threads
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Stop components in the correct order
            
            # 1. First stop feature extraction (audio capture)
            if self.feature_extractor:
                logger.info("Stopping feature extraction...")
                self.feature_extractor.stop()
            
            # 2. Then stop frequency generator (audio output)
            if self.freq_generator:
                logger.info("Stopping frequency generator...")
                self.freq_generator.stop()
            
            # 3. Stop GUI
            if self.gui:
                logger.info("Stopping GUI...")
                self.gui.stop()
            
            # 4. Close logger
            logger.info("Closing logger...")
            close_logger()
            
            # Wait for threads to finish
            for name, thread in self.threads.items():
                if thread.is_alive():
                    logger.info(f"Waiting for {name} thread to finish...")
                    thread.join(timeout=2.0)
            
            logger.info("System stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            return False
    
    def prediction_loop(self):
        """
        Main prediction loop that runs in a separate thread.
        
        Continuously gets features, makes predictions, and updates GUI.
        """
        # Give feature extraction time to start producing data
        time.sleep(2.0)
        
        last_prediction_time = 0
        prediction_interval = 0.1  # Make predictions every 100ms
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check if it's time for a new prediction
                now = time.time()
                if now - last_prediction_time >= prediction_interval:
                    # Get latest features
                    features_dict = self.feature_extractor.get_latest_features(timeout=0.2)
                    
                    if features_dict:
                        # Prepare features for model
                        scaled_features = self.feature_extractor.prepare_features_for_model(
                            features_dict, self.scaler
                        )
                        
                        # Validate features
                        if validate_features(scaled_features):
                            # Make prediction
                            position, curvature = predict(self.model, scaled_features)
                            
                            # Process prediction
                            processed_position, processed_curvature, is_active = process_prediction(
                                position, curvature
                            )
                            
                            # Update GUI
                            self.gui.update_values(processed_curvature, processed_position, is_active)
                            
                            # Log prediction
                            from rt_logger import log_data
                            log_data(processed_curvature, processed_position)
                            
                            # Update last prediction time
                            last_prediction_time = now
                    
                # Short sleep to prevent CPU overuse
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(0.1)  # Sleep longer on error
    
    def run(self):
        """Run the system until interrupted by user."""
        if not self.start():
            logger.error("Failed to start system")
            return False
        
        try:
            logger.info("System running. Starting GUI on main thread...")
            
            # Run GUI on the main thread - this will block until GUI is closed
            self.gui.start()
            
        except KeyboardInterrupt:
            logger.info("User interrupted execution")
        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
        finally:
            self.stop()

        return True


def main():
    """Main entry point for the real-time prediction system."""
    logger.info("Starting real-time curvature prediction system")
    
    # Create and run orchestrator
    orchestrator = Orchestrator()
    success = orchestrator.run()
    
    if success:
        logger.info("System completed successfully")
        return 0
    else:
        logger.error("System exited with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())