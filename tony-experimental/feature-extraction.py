# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read 
from IPython.display import Audio
from numpy.fft import fft, ifft
import pandas as pd

# %% Read the metadata
metaData = pd.read_csv('../data/UrbanSound8K/metadata/UrbanSound8K.csv')

# %% Read audio input (1 file for now)
sampleRate, soundData = read('../data/UrbanSound8K/audio/fold1/7061-6-0-0.wav')
sDataMono = soundData[:,0]
clipLength = soundData.shape[0] / sampleRate
print(f'Sample rate = {sampleRate}')
print(f'Data points = {soundData.shape[0]}')
print(f'Number of channels = {soundData.shape[1]}')
print(f'Length = {clipLength}s')

# %% Play Audio
Audio(sDataMono, rate=sampleRate)

# %% Plot the waveform (mono)
plt.figure()
plt.plot(sDataMono)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Waveform of Audio')
plt.show

# %%
# Plot the waveform (stereo)
time = np.linspace(0, clipLength, soundData.shape[0])
plt.plot(time, soundData[:, 0], label="Left channel")
plt.plot(time, soundData[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()