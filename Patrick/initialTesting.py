# %% Imports

#general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import os

#audio processing/tools import
import librosa
import librosa.display
from scipy.io.wavfile import read 
from IPython.display import Audio

#unknown imports
import glob 

# %% Read the metadata
metaData = pd.read_csv('../.data/UrbanSound8K/metadata/UrbanSound8K.csv')

#metadata read functions

def PatrickAudioClass(filename):
   return    metaData.loc[metaData['slice_file_name'] == filename].classID

# %% Read audio input (19026-1-0-0.wav - car horn, classID = 1)

files = glob.glob('../.data/UrbanSound8K/audio/patrickTemp/*.wav')

#audioFile, sampling_rate = librosa.load('../.data/UrbanSound8K/audio/patrickTemp/19026-1-0-0.wav')

for i in files:
    files[i] = os.path.basename(files[i])

files


# %% Plot librosa audio visualizations
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)

plt.figure(figsize=(12, 4))
librosa.display.specshow(data)

# %% Feature Engineering with Librosa

#Mel-frequency cepstral coefficients (MFCCs)
#Yeah idk what that is but it seems legit ^

mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# %%


# %%
