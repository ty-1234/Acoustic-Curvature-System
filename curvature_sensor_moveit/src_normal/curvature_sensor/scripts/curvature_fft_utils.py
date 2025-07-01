
import numpy as np

def extract_fft_features(signal, sample_rate=22050, window_size=10000, target_freqs=None):
    """
    Perform FFT on a 1D audio signal window and extract amplitudes at target frequencies.
    
    This function applies the Fast Fourier Transform to the input signal and extracts 
    the amplitude values at specified target frequencies. It's useful for identifying
    frequency components in audio or sensor signals.
    
    Parameters
    ----------
    signal : np.array
        1D array containing the time-domain signal data
    sample_rate : int, optional
        Sampling rate of the signal in Hz (default: 22050)
    window_size : int, optional
        Size of the FFT window in samples (default: 10000)
    target_freqs : list or None, optional
        List of frequencies (in Hz) to extract. If None, defaults to frequencies 
        from 200 Hz to 2000 Hz in steps of 200 Hz
    
    Returns
    -------
    list
        List of FFT amplitude values corresponding to each target frequency
        
    Notes
    -----
    The function performs DC correction on the first frequency component.
    
    Example
    -------
    >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    >>> extract_fft_features(signal, target_freqs=[440, 880])
    [0.9998..., 0.0001...]
    """
    if target_freqs is None:
        target_freqs = list(range(200, 2001, 200))  # Default: 200 Hz to 2000 Hz

    Fn = sample_rate / 2
    fft_result = np.fft.fft(signal, window_size) / window_size
    freqs = np.linspace(0, Fn, int(window_size / 2) + 1)
    magnitudes = 2 * np.abs(fft_result[:int(window_size / 2) + 1])
    magnitudes[0] /= 2  # DC correction

    fft_values = []
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        fft_values.append(magnitudes[idx])

    return fft_values
