# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read 
from IPython.display import Audio
from numpy.fft import fft, ifft
import pandas as pd

# %% Read the metadata
metaData = pd.read_csv('../data/UrbanSound8K/metadata/UrbanSound8K.csv')
classes = list(metaData['class'].unique()) 

# %% Load functions

# Function to plot the waveform (stereo)
def plt_orig_waveform(sampleRate, soundData, channels):
    if channels == 'mono':
        soundData = soundData[:,0]
    clipLength = soundData.shape[0] / sampleRate
    time = np.linspace(0, clipLength, soundData.shape[0])
    plt.plot(time, soundData[:, 0], label="Left channel")
    plt.plot(time, soundData[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    print(f'Sample rate = {sampleRate}')
    print(f'Data points = {soundData.shape[0]}')
    print(f'Number of channels = {soundData.shape[1]}')
    print(f'Length = {clipLength}s')

# %% Grab one audio file from each class
files = dict()
for i in range(len(classes)):
    tmp = metaData[metaData['class'] == classes[i]][:1].reset_index()
    path = (
            '../data/UrbanSound8K/audio/fold{}/{}'
            .format(tmp['fold'][0], tmp['slice_file_name'][0])
            )
    files[classes[i]] = path

# %% Plot the waveforms
fig = plt.figure(figsize=(15,15))# Log graphic of waveforms to Comet
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, label in enumerate(classes):
    fn = files[label]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    sampleRate, soundData = read(fn) # alt: soundData, sampleRate = librosa.load(fn)
    Audio(soundData[:,0], rate=sampleRate) # IPython play audio widget
    plt_orig_waveform(sampleRate, soundData, 'stereo')
    # alt: librosa.display.waveplot(data, sr= sample_rate)
plt.savefig('class_examples.png')

# %%
for x in range(1250,1500):
    print(soundData[x,0])

# %%
# https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/