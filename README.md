Natural-Sampling

Aim :

To perform Natural Sampling of a continuous-time message signal using a pulse train and reconstruct the signal using a low-pass filter.

Tools required :

Python (Version 3.x)

Numpy Library (for numerical computation)

Matplotlib Library (for plotting the waveforms)

Scipy Library (for signal processing, including filtering)

Program :

#Natural sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000 # Sampling frequency (samples per second)

T = 1 # Duration in seconds

t = np.arange(0, T, 1/fs) # Time vector

fm = 5 # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50 # pulses per second

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):

pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):

index = np.argmin(np.abs(t - time))

reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
w-pass Filter (optional, smoother reconstruction)

def lowpass_filter(signal, cutoff, fs, order=5):

nyquist = 0.5 * fs

normal_cutoff = cutoff / nyquist

b, a = butter(order, normal_cutoff, btype='low', analog=False)

return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(t, nat_signal, label='Natural Sampling')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

Output Waveform :

![image](https://github.com/user-attachments/assets/102b4314-3b47-4779-8617-e3a1b2eee622)
![image](https://github.com/user-attachments/assets/108c0634-9235-4c8f-a0f7-e255ec3d398b)
![image](https://github.com/user-attachments/assets/b92c1b3e-8877-4351-913f-ece5b39d32e5)
![image](https://github.com/user-attachments/assets/76206e4b-7477-40c7-87e1-606379e83f0d)

Results :

Original Message Signal:
A 5 Hz sine wave is generated as the input message signal.

Pulse Train Generation:
A pulse train with a pulse rate of 50 Hz is created using rectangular pulses.

Natural Sampling:
The message signal is multiplied by the pulse train, producing the sampled waveform.

Reconstruction Process:
The sampled signal undergoes zero-order hold interpolation to reconstruct the original message.

A low-pass Butterworth filter is applied for smooth reconstruction.
