import numpy as np

def extract_fft_features(signal, sample_rate=22050, window_size=10000, target_freqs=None):
    """
    Perform FFT on a 1D audio signal window and extract amplitudes at target frequencies.

    Args:
        signal (np.array): 1D audio data (single window)
        sample_rate (int): Sampling rate in Hz (default: 22050)
        window_size (int): Length of FFT window (default: 10000)
        target_freqs (list or None): List of frequencies to extract. If None, defaults to 200–2000 Hz (step 200)

    Returns:
        List of FFT amplitudes at the target frequencies
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
