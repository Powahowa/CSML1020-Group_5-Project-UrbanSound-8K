# General stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import os
import pathlib

# Audio processing/tools import
import librosa
import librosa.display
from scipy.io.wavfile import read 
from IPython.display import Audio
#REMEMBER you need ffmpeg installed

# other imports
import glob #filesystem manipulation

# Define some decorator functions
import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__!r}')
        startTime = time.perf_counter()
        value = func(*args, **kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f'Finished {func.__name__!r} in {runTime:.4f} secs')
        return value
    return wrapper

@timer
def init_data(relPathToFolder):
    # Read in the metadata
    metaData = pd.read_csv(relPathToFolder + 'metadata/UrbanSound8K.csv')

    #recursively add all .wave files to paths list
    paths = list(pathlib.Path(relPathToFolder + 'audio/').glob('**/*.wav'))
    fileNames = paths.copy()

    #remove path from fileNames leaving us just with the raw filename
    for i in range(len(fileNames)):
        fileNames[i] = os.path.basename(fileNames[i])

    #create dataframe from paths and filenames
    fileData = pd.DataFrame(list(zip(paths, fileNames)), 
                            columns =['path', 'slice_file_name']) 

    #merge metadata and fileData (the one we just created) dataframes
    fileData = fileData.join(metaData.set_index('slice_file_name'), 
                             on='slice_file_name')
    return fileData

@timer
def init_metadata(relPathToFolder):
    # Read in the metadata
    metaData = pd.read_csv(relPathToFolder + 'metadata/UrbanSound8K.csv')
    tmp = []
    for i in range(len(metaData)):
        fold = metaData['fold'][i]
        slice_file_name = metaData['slice_file_name'][i]
        tmp.append(relPathToFolder + 
            f'audio/fold{fold}/{slice_file_name}')
    metaData = metaData.join(pd.DataFrame(tmp))
    return metaData

@timer
def test_read_audio(filepath):
    audioFile, samplingRate = load_audio(filepath)

    # Plot librosa audio visualizations
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(audioFile, sr=samplingRate)

    #plt.figure(figsize=(12, 4))
    #librosa.display.specshow(audioFile)

    # Feature Engineering with Librosa

    #Mel-frequency cepstral coefficients (MFCCs)

    mfccs = librosa.feature.mfcc(y=audioFile, sr=samplingRate,  n_mfcc=40)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

@timer
def mfccsEngineering(filepath):
      audioFile, sampling_rate = load_audio(filepath)
      mfccs = librosa.feature.mfcc(y=audioFile, sr=sampling_rate,  n_mfcc=40)
      mfccs = np.mean(mfccs.T,axis=0)
      return mfccs

# Function to plot the waveform (stereo)
@timer
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

@timer
def load_audio(filepaths):
    sounds = []
    rates = []
    for f in filepaths:
        y, sr = librosa.load(f, sr=None)
        sounds.append(y)
        rates.append(sr)
    return sounds, rates