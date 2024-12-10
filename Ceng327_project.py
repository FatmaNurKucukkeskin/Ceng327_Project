import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq

# Load the audio file
audio_path = '1.wav'  # Add the audio file you want to analyze here
y, sr = librosa.load(audio_path, sr=None)
print(f"Sampling Rate: {sr} Hz")
print(f"Frame Count: {len(y)}")

# 1. Time Domain Visualization
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# 2. RMS and dB Calculation
rms = np.sqrt(np.mean(y**2))  # Calculate RMS
rms_db = 20 * np.log10(rms)  # Convert to Decibels (dB)
print(f"RMS Value: {rms:.5f}")
print(f"Sound Level (dB): {rms_db:.2f}")

# 3. Frequency Spectrum Analysis
fft_result = fft(y)  # Compute FFT
frequencies = fftfreq(len(y), 1 / sr)  # Frequency axis
pos_mask = frequencies >= 0  # Take only positive frequencies
fft_magnitude = np.abs(fft_result[pos_mask])
fft_frequencies = frequencies[pos_mask]

plt.figure(figsize=(10, 4))
plt.plot(fft_frequencies, fft_magnitude)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sr // 2)  # Limit to human hearing range
plt.show()

# 4. Perceptual Analysis: A-Weighting Filter
def a_weighting_filter(frequencies):
    """
    A-Weighting filter calculation.
    """
    f = frequencies
    Ra = (12194**2 * f**4) / ((f**2 + 20.6**2) * ((f**2 + 107.7**2)**0.5) * ((f**2 + 737.9**2)**0.5) * (f**2 + 12194**2))
    A = 20 * np.log10(Ra) - 20 * np.log10(12194**2 / ((20.6**2) * (12194**2)))
    return A

a_weight = a_weighting_filter(fft_frequencies)

# Apply A-Weighting
fft_magnitude_weighted = fft_magnitude + a_weight

plt.figure(figsize=(10, 4))
plt.plot(fft_frequencies, fft_magnitude_weighted)
plt.title("A-Weighted Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weighted Magnitude (dB)")
plt.xlim(0, sr // 2)
plt.show()

# 5. Print Results
print("\nPhysical and Perceptual Analysis Completed!")
print(f"Decibel (dB) RMS: {rms_db:.2f}")
print(f"A-Weighting Average Value: {np.mean(a_weight):.2f} dB")

# Additional Information for the User
print("\nThe frequency spectrum has been weighted according to human perception.")
