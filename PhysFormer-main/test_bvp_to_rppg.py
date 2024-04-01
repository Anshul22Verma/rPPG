import numpy as np
import matplotlib.pyplot as plt

# Sample BVP data
BVP_data = np.array([0.1, 0.2, 0.3, 0.6, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1])

# Find peaks (simplified, assuming peaks are where BVP signal is above a threshold)
peaks_indices = np.where(BVP_data > 0.5)[0]
peak_amplitudes = BVP_data[peaks_indices]

# Construct RPPG signal (for simplicity, we'll directly use peak amplitudes)
RPPG_signal = peak_amplitudes

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(BVP_data, label='BVP')
plt.scatter(peaks_indices, peak_amplitudes, color='red', label='Peaks')
plt.plot(RPPG_signal, color='green', label='RPPG')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('BVP and RPPG Signals')
plt.legend()
plt.grid(True)
plt.show()
