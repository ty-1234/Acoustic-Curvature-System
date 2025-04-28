"""
Real-time multi-tone frequency generator for curvature sensing experiments.

This script generates and plays a continuous multi-tone signal containing sine waves
at frequencies from 200 Hz to 2000 Hz (in steps of 200 Hz). It provides a consistent
acoustic input signal for real-time curvature sensing experiments.

Author: Bipindra Rai
Date: 2025-04-28
"""

import numpy as np
import sounddevice as sd
import threading
import time
import logging
import queue
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

class RealTimeFrequencyGenerator:
    def __init__(self):
        """Initialize the real-time frequency generator with audio settings."""
        # Audio parameters
        self.sample_rate = 44100  # Hz
        self.block_size = 4410    # 100ms blocks
        self.frequencies = list(range(200, 2001, 200))  # [200, 400, ..., 2000]
        
        # Control variables
        self.running = False
        self.audio_thread = None
        self.buffer_queue = queue.Queue(maxsize=10)  # Buffer for audio blocks
        
        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info(f"🎵 Frequency generator initialized with frequencies: {self.frequencies} Hz")

    def _generate_block(self, t_start):
        """
        Generate a block of multi-tone audio data.
        
        Parameters:
        -----------
        t_start : float
            Starting time point for this audio block
            
        Returns:
        --------
        np.ndarray
            Audio data as float32 array normalized to [-1.0, 1.0]
        """
        # Generate time points for this block
        t = np.linspace(
            t_start, 
            t_start + self.block_size / self.sample_rate, 
            self.block_size, 
            endpoint=False
        )
        
        # Generate multi-tone signal
        signal = np.zeros_like(t, dtype=np.float32)
        for freq in self.frequencies:
            # Scale amplitude to prevent clipping when combined
            scale_factor = 0.8 / len(self.frequencies)
            signal += scale_factor * np.sin(2 * np.pi * freq * t)
        
        return signal
    
    def _audio_callback(self, outdata, frames, time, status):
        """
        Callback function for the sounddevice output stream.
        
        This is called by sounddevice when it needs more audio data to play.
        It retrieves pre-generated audio blocks from the queue.
        
        Parameters:
        -----------
        outdata : ndarray
            Output buffer to fill with audio data
        frames : int
            Number of frames to generate
        time : CData
            Timestamp information
        status : CallbackFlags
            Status flags indicating if there were errors
        """
        if status:
            logging.warning(f"⚠️ Audio output status: {status}")
        
        try:
            # Get next audio block from queue
            data = self.buffer_queue.get_nowait()
            outdata[:] = data.reshape(-1, 1)  # Reshape to match outdata dimensions
        except queue.Empty:
            # If queue is empty, generate silence
            logging.warning("⚠️ Buffer underrun - audio queue empty")
            outdata.fill(0)
    
    def _generate_audio_blocks(self):
        """
        Continuously generate audio blocks and add them to the queue.
        
        This runs in a separate thread to keep the audio buffer filled.
        """
        t_pos = 0.0  # Starting time position
        
        try:
            while self.running:
                # Generate next audio block
                audio_block = self._generate_block(t_pos)
                
                # Update time position for next block
                t_pos += self.block_size / self.sample_rate
                
                # Add to queue (with timeout to allow checking running flag)
                try:
                    self.buffer_queue.put(audio_block, timeout=1.0)
                except queue.Full:
                    # Skip this block if queue is full
                    pass
                
                # Small sleep to prevent CPU burn
                time.sleep(0.01)
                
        except Exception as e:
            logging.error(f"❌ Error in audio generation thread: {e}")
            self.running = False
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals by stopping the generator."""
        logging.info("\n🛑 Received termination signal")
        self.stop()
        sys.exit(0)
            
    def start(self):
        """
        Start the real-time frequency generator.
        
        Returns:
        --------
        bool
            True if started successfully, False otherwise
        """
        if self.running:
            logging.warning("⚠️ Frequency generator already running")
            return True
        
        try:
            # Set running flag
            self.running = True
            
            # Start the audio generation thread
            self.audio_thread = threading.Thread(
                target=self._generate_audio_blocks,
                daemon=True
            )
            self.audio_thread.start()
            
            # Pre-fill the buffer queue
            start_time = time.time()
            while self.buffer_queue.qsize() < 5 and time.time() - start_time < 2.0:
                time.sleep(0.1)
            
            # Open the audio output stream
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.audio_stream.start()
            
            logging.info("✅ Frequency generator started successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to start frequency generator: {e}")
            self.running = False
            return False
    
    def stop(self):
        """
        Stop the real-time frequency generator.
        
        Returns:
        --------
        bool
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logging.warning("⚠️ Frequency generator not running")
            return True
        
        try:
            # Set running flag to stop generation thread
            self.running = False
            
            # Stop and close the audio stream
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Wait for audio thread to finish
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            # Clear the buffer queue
            while not self.buffer_queue.empty():
                try:
                    self.buffer_queue.get_nowait()
                except queue.Empty:
                    break
            
            logging.info("✅ Frequency generator stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error stopping frequency generator: {e}")
            return False
    
    def is_running(self):
        """
        Check if the frequency generator is currently running.
        
        Returns:
        --------
        bool
            True if running, False otherwise
        """
        return self.running


def main():
    """
    Main function that initializes and starts the frequency generator.
    
    This function serves as a standalone entry point and demonstration
    of the generator's capabilities.
    """
    generator = RealTimeFrequencyGenerator()
    
    try:
        # Start the generator
        if generator.start():
            logging.info("🎵 Playing multi-tone signal. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while generator.is_running():
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logging.info("\n🛑 Frequency generator stopped by user")
    
    finally:
        # Ensure generator is stopped
        generator.stop()
        logging.info("👋 Frequency generator exited")


if __name__ == "__main__":
    main()