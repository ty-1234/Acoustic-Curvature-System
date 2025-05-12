"""
Real-time GUI for curvature sensor predictions.

This module provides a Tkinter-based GUI that displays the most recent
curvature and position predictions in real-time with continuous updates.

Author: Bipindra Rai
Date: 2025-04-28
"""

import tkinter as tk
import time
import logging
import threading

# Set up logging
logger = logging.getLogger(__name__)

class PredictionGUI:
    """
    Graphical user interface for displaying real-time curvature and position predictions.
    
    This class creates a Tkinter window that displays the latest prediction values
    and updates them continuously without requiring user interaction.
    """
    
    def __init__(self, update_interval=100):
        """
        Initialize the prediction GUI.
        
        Parameters:
        -----------
        update_interval : int
            How often to check for new values (in milliseconds)
        """
        self.update_interval = update_interval
        
        # Latest prediction values
        self.current_curvature = 0.0
        self.current_position = -1.0  # -1 indicates inactive
        self.is_active = False
        
        # Microphone information
        self.mic_name = "Unknown"
        
        # Initialize GUI components
        self.root = None
        self.label_title = None
        self.label_curvature = None
        self.label_position = None
        self.label_status = None
        self.label_mic = None  # New label for microphone info
        self.frame_values = None
        
        # For window closing
        self.is_running = False
        
        # Thread safety for updates
        self.update_lock = threading.Lock()
        
        logger.info("Prediction GUI initialized")
    
    def update_values(self, curvature, position, is_active=None):
        """
        Update the prediction values to be displayed.
        
        Parameters:
        -----------
        curvature : float
            Current curvature prediction (mm^-1)
        position : float
            Current position prediction (cm)
        is_active : bool or None
            Whether the sensor is currently active
        """
        with self.update_lock:
            self.current_curvature = curvature
            self.current_position = position
            
            # If is_active is not provided, infer from position
            if is_active is None:
                self.is_active = position > 0
            else:
                self.is_active = is_active
    
    def update_mic_info(self, mic_name):
        """
        Update the microphone information displayed in the GUI.
        
        Parameters:
        -----------
        mic_name : str
            Name of the current microphone device
        """
        with self.update_lock:
            self.mic_name = mic_name
            
            # Update the label if it exists
            if hasattr(self, 'label_mic') and self.label_mic:
                self.label_mic.config(text=f"🎤 Microphone: {self.mic_name}")
    
    def _update_display(self):
        """
        Update the GUI labels with the latest values.
        """
        if not self.is_running:
            return
            
        with self.update_lock:
            # Format the values for display
            curvature_str = f"{self.current_curvature:.4f}" if self.is_active else "N/A"
            position_str = f"{self.current_position:.1f}" if self.is_active else "N/A"
            
            # Update the values with appropriate styling based on active state
            self.label_curvature.config(
                text=f"Curvature: {curvature_str} mm⁻¹", 
                fg="#0066CC" if self.is_active else "gray"
            )
            self.label_position.config(
                text=f"Position: {position_str} cm", 
                fg="#0066CC" if self.is_active else "gray"
            )
            
            # Update status indicator
            status_text = "ACTIVE" if self.is_active else "INACTIVE"
            status_color = "green" if self.is_active else "red"
            self.label_status.config(text=status_text, fg=status_color)
        
        # Schedule next update
        if self.root:
            self.root.after(self.update_interval, self._update_display)
    
    def start(self):
        """
        Start the GUI and enter the main event loop.
        
        This method blocks until the window is closed.
        """
        if self.is_running:
            logger.warning("GUI is already running")
            return
            
        self.is_running = True
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Curvature Sensor Predictions")
        self.root.geometry("400x350")  # Increased height for mic info
        self.root.resizable(False, False)
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap("sensor_icon.ico")  # Replace with actual icon path if available
        except:
            pass
        
        # Configure window close event
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        # Create title label
        self.label_title = tk.Label(
            self.root, 
            text="Real-Time Curvature Sensor",
            font=("Helvetica", 16, "bold"),
            pady=15
        )
        self.label_title.pack()
        
        # Create microphone info label with consistent styling
        self.label_mic = tk.Label(
            self.root,
            text=f"🎤 Microphone: {self.mic_name}",
            font=("Helvetica", 11, "bold"),
            fg="#0066CC",
            pady=8,
            bg=self.root.cget('bg')
        )
        self.label_mic.pack()
        
        # Create status indicator
        self.label_status = tk.Label(
            self.root, 
            text="INACTIVE",
            font=("Helvetica", 12, "bold"),
            fg="red",
            pady=5
        )
        self.label_status.pack()
        
        # Create frame for values with appropriate padding
        self.frame_values = tk.Frame(self.root, pady=20)
        self.frame_values.pack(fill=tk.X, padx=20)
        
        # Create value labels with high visibility styling
        self.label_curvature = tk.Label(
            self.frame_values, 
            text="Curvature: N/A mm⁻¹",
            font=("Helvetica", 14, "bold"),
            pady=10,
            fg="#0066CC"  # Blue text for better visibility
        )
        self.label_curvature.pack()

        self.label_position = tk.Label(
            self.frame_values, 
            text="Position: N/A cm",
            font=("Helvetica", 14, "bold"),
            pady=10,
            fg="#0066CC"  # Blue text for better visibility
        )
        self.label_position.pack()
        
        # Add timestamp at bottom
        timestamp_label = tk.Label(
            self.root,
            text=f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            font=("Helvetica", 10),
            fg="gray",
            pady=10
        )
        timestamp_label.pack(side=tk.BOTTOM)
        
        # Start the update cycle
        self.root.after(self.update_interval, self._update_display)
        
        logger.info("Prediction GUI started")
        
        # Enter the Tkinter event loop
        self.root.mainloop()
    
    def stop(self):
        """
        Stop the GUI and close the window.
        """
        self.is_running = False
        
        if self.root:
            self.root.destroy()
            self.root = None
            
        logger.info("Prediction GUI stopped")
    
    def is_gui_active(self):
        """
        Check if the GUI is still running.
        
        Returns:
        --------
        bool
            True if the GUI is still active and displayed
        """
        return self.is_running and self.root is not None

# For quick testing
def run_test_gui():
    """
    Run a test of the GUI with simulated changing values.
    """
    import random
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create the GUI instance
    gui = PredictionGUI()
    
    # Start in a separate thread so we can update values
    gui_thread = threading.Thread(target=gui.start)
    gui_thread.daemon = True
    gui_thread.start()
    
    try:
        # Simulate sensor being inactive initially
        gui.update_values(0, -1, False)
        time.sleep(2)
        
        # Simulate sensor becoming active and getting varying readings
        for i in range(100):
            # Simulate some activity with realistic values
            is_active = True if i > 5 else False
            
            if is_active:
                curvature = 0.05 + 0.01 * random.random()  # 0.05-0.06 mm^-1
                position = 5.0 + random.random() * 2  # 5-7 cm
            else:
                curvature = 0
                position = -1
                
            # Update the GUI
            gui.update_values(curvature, position, is_active)
            
            # Wait a bit
            time.sleep(0.2)
            
        # Simulate sensor becoming inactive again
        gui.update_values(0, -1, False)
        time.sleep(2)
        
    except KeyboardInterrupt:
        pass
    finally:
        # Let the GUI run for a bit longer before ending the test
        time.sleep(3)
        gui.stop()

# Run test if the script is executed directly
if __name__ == "__main__":
    run_test_gui()