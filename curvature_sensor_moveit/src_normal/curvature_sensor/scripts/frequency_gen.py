
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
        # Example for higher sample rate (check hardware support if going above Nyquist):
        # self.sample_rate = 96000 # Hz (max representable freq: 48kHz)
        
        self.block_size = 4410    # Number of frames per block (e.g., 100ms at 44.1kHz)
                                  # If changing sample_rate, you might want to adjust block_size
                                  # to maintain similar latency, e.g., self.block_size = int(0.1 * self.sample_rate)

        # --- MODIFIED FREQUENCY RANGE ---
        self.frequencies = list(range(100, 16001, 100))  # [100, 200, ..., 16000]
        
        # --- NYQUIST FREQUENCY WARNING (still useful if sample_rate is changed or frequencies go higher) ---
        self.nyquist_freq = self.sample_rate / 2
        if max(self.frequencies) > self.nyquist_freq:
            logging.warning(
                f"🚨 WARNING: Maximum requested frequency ({max(self.frequencies)} Hz) "
                f"exceeds Nyquist frequency ({self.nyquist_freq} Hz) for the current "
                f"sample rate ({self.sample_rate} Hz). Frequencies above {self.nyquist_freq} Hz "
                f"will be aliased (distorted). Consider increasing self.sample_rate if applicable."
            )
        
        # Control variables
        self.running = False
        self.audio_thread = None
        self.buffer_queue = queue.Queue(maxsize=10)  # Buffer for audio blocks
        
        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info(f"🎵 Frequency generator initialized. Target frequencies: {min(self.frequencies)} Hz to {max(self.frequencies)} Hz.")
        logging.info(f"   Sample Rate: {self.sample_rate} Hz. Nyquist Frequency: {self.nyquist_freq} Hz.")
        logging.info(f"   Number of tones: {len(self.frequencies)}.")
        logging.info("   Ensure your audio output device (e.g., speaker/transducer) can reproduce the desired frequency range.")


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
        combined_signal = np.zeros_like(t, dtype=np.float32)
        if not self.frequencies: 
            return combined_signal

        # Scale amplitude to prevent clipping when combined.
        # 0.8 is used to leave some headroom.
        scale_factor = 0.8 / len(self.frequencies) 
        
        for freq in self.frequencies:
            combined_signal += scale_factor * np.sin(2 * np.pi * freq * t)
        
        return combined_signal
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Callback function for the sounddevice output stream.
        Retrieves pre-generated audio blocks from the queue.
        """
        if status:
            logging.warning(f"⚠️ Audio output status: {status}")
        
        try:
            data = self.buffer_queue.get_nowait()
            outdata[:] = data.reshape(-1, 1)
        except queue.Empty:
            logging.warning("⚠️ Buffer underrun - audio queue empty. Outputting silence.")
            outdata.fill(0)
    
    def _generate_audio_blocks(self):
        """
        Continuously generate audio blocks and add them to the queue.
        This runs in a separate thread.
        """
        t_current_block_start = 0.0
        
        try:
            while self.running:
                if self.buffer_queue.qsize() < self.buffer_queue.maxsize -1: 
                    audio_block = self._generate_block(t_current_block_start)
                    t_current_block_start += self.block_size / self.sample_rate
                    try:
                        self.buffer_queue.put(audio_block, timeout=0.05)
                    except queue.Full:
                        logging.debug("Audio buffer queue full, generation paused briefly.")
                        time.sleep(0.01) 
                else:
                    time.sleep(self.block_size / (2 * self.sample_rate) ) 
                
        except Exception as e:
            logging.error(f"❌ Error in audio generation thread: {e}", exc_info=True)
            self.running = False 
    
    def _signal_handler(self, sig, frame):
        logging.info(f"\n🛑 Received signal {sig}. Attempting graceful shutdown...")
        self.stop()
            
    def start(self):
        """
        Start the real-time frequency generator.
        """
        if self.running:
            logging.warning("⚠️ Frequency generator already running.")
            return True
        
        try:
            self.running = True
            while not self.buffer_queue.empty():
                try:
                    self.buffer_queue.get_nowait()
                except queue.Empty:
                    break

            self.audio_thread = threading.Thread(
                target=self._generate_audio_blocks,
                daemon=True 
            )
            self.audio_thread.start()
            
            prefill_target = min(5, self.buffer_queue.maxsize -1) 
            start_time = time.time()
            while self.buffer_queue.qsize() < prefill_target and (time.time() - start_time < 2.0) and self.running:
                time.sleep(0.05) 
            
            if not self.running: 
                logging.info("Generator stopped during prefill.")
                if self.audio_thread.is_alive(): self.audio_thread.join(timeout=1.0)
                return False

            if self.buffer_queue.qsize() < prefill_target:
                logging.warning(f"⚠️ Buffer prefill incomplete ({self.buffer_queue.qsize()}/{prefill_target} blocks). Starting stream anyway.")

            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1, 
                callback=self._audio_callback,
                dtype='float32'
            )
            self.audio_stream.start()
            
            logging.info("✅ Frequency generator started successfully.")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to start frequency generator: {e}", exc_info=True)
            self.stop() 
            return False
    
    def stop(self):
        """
        Stop the real-time frequency generator.
        """
        if not self.running and not (hasattr(self, 'audio_stream') and self.audio_stream):
            logging.info("ℹ️ Frequency generator appears to be already stopped or was not started.")
            return True
        
        logging.info("Stopping frequency generator...")
        self.running = False 
        
        try:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                if not self.audio_stream.stopped:
                    self.audio_stream.stop()
                if not self.audio_stream.closed:
                    self.audio_stream.close()
                self.audio_stream = None 
                logging.debug("Audio stream stopped and closed.")
            
            if self.audio_thread and self.audio_thread.is_alive():
                logging.debug("Waiting for audio generation thread to finish...")
                self.audio_thread.join(timeout=2.0) 
                if self.audio_thread.is_alive():
                    logging.warning("⚠️ Audio generation thread did not finish in time.")
                else:
                    logging.debug("Audio generation thread finished.")
            
            logging.debug("Clearing audio buffer queue...")
            while not self.buffer_queue.empty():
                try:
                    self.buffer_queue.get_nowait()
                except queue.Empty:
                    break
            logging.debug("Audio buffer queue cleared.")
            
            logging.info("✅ Frequency generator stopped successfully.")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error stopping frequency generator: {e}", exc_info=True)
            return False
    
    def is_running(self):
        return self.running


def main():
    main_stop_event = threading.Event()
    def main_signal_handler(sig, frame):
        logging.info(f"\n🛑 Main received signal {sig}, setting stop event.")
        main_stop_event.set()

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, main_signal_handler)
    signal.signal(signal.SIGTERM, main_signal_handler)

    generator = RealTimeFrequencyGenerator()
    
    try:
        if generator.start():
            logging.info("🎵 Playing multi-tone signal. Press Ctrl+C to stop.")
            while generator.is_running() and not main_stop_event.is_set():
                time.sleep(0.1) 
        else:
            logging.error("Could not start the frequency generator.")
            
    except Exception as e: 
        logging.error(f"❌ An unexpected error occurred in main: {e}", exc_info=True)

    finally:
        logging.info("Initiating final cleanup...")
        if hasattr(generator, 'is_running') and generator.is_running(): 
            generator.stop()
        
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        logging.info("👋 Frequency generator exited.")


if __name__ == "__main__":
    main()
