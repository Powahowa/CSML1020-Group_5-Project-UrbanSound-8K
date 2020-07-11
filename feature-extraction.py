# %% [markdown]
# # Feature Extraction
# ## Imports

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
# REMEMBER you need ffmpeg installed

# Parellization libraries
from joblib import Parallel, delayed

# Import custom module containing useful functions
import sonicboom

# Import helper functions
import time

# %% [markdown]
# ## Define some constants
SAVEPATH = './output/intermediate-data/'
FILEDESC = 'filedata-librosaFeatures.pkl'

# %% [markdown]
# ## Read and add filepaths to original UrbanSound metadata
filedata = sonicboom.init_data('./data/UrbanSound8K/')

# %% [markdown]
# ## Sample down

sampleDown = False

# samples down grouping by class - this gives me X items from each class.
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
if (sampleDown == True):
    filedata = filedata.groupby(
        'class', 
        as_index=False, 
        group_keys=False
    ).apply(lambda x: x.sample(n=2, random_state=0))

# check that the sample down is working
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
filedata.groupby('class', as_index=False)['slice_file_name'].nunique()

# %% [markdown]
# ## Read one audio file to see what it contains
sonicboom.test_read_audio(filedata.path.iloc[0])

# %%
# Get sample rates for all .wave files and add to filedata dataframe

filedata['Sample Rate'] = Parallel(n_jobs=-1)(delayed(sonicboom.samplerate)(x) for x in filedata['path'])

filedata.to_csv('./SampleRates.csv')

# %% [markdown]
# ## PARALLEL Generate features and add to dataframe
startTime = time.perf_counter()

#non-parallel version
#filedata['mfccs'] = [sonicboom.mfccsEngineering(x) for x in filedata['path']]

# inputVar = input("0. All, \n \
#     1. MFCCS \n \
#     2. Mel-scaled spectrogram \n \
#     3. Short-time Fourier transform (STFT) \n \
#     4. Chromagram (STFT) \n \
#     5. Spectral contrast (STFT) \n \
#     6. Tonal centroid features (tonnetz) from harmonic components \n")

mfccs_exec = True
melSpec_exec = True
stft_exec = True
chroma_stft_exec = True
spectral_contrast_stft_exec = True
tonnetz_exec = True

if (mfccs_exec == True):
    #generate mfccs features
    filedata['mfccs'] = Parallel(n_jobs=-1)(delayed(sonicboom.mfccsEngineering)(x) for x in filedata['path'])
    print("MFCCS done!")
if (melSpec_exec == True):
    #generate melSpec features
    filedata['melSpec'] = Parallel(n_jobs=-1)(delayed(sonicboom.melSpecEngineering)(x) for x in filedata['path'])
    print("Mel-scaled spectrogram done!")

if (stft_exec == True):
    #generate stft features
    filedata['stft'] = Parallel(n_jobs=-1)(delayed(sonicboom.stftEngineering)(x) for x in filedata['path'])
    print("Short-time Fourier transform (STFT) done!")

if (chroma_stft_exec == True):
    #generate chroma_stft features
    filedata['chroma_stft'] = Parallel(n_jobs=-1)(delayed(sonicboom.chroma_stftEngineering)(x) for x in filedata['path'])
    print("Chromagram (STFT) done!")

if (spectral_contrast_stft_exec == True):
    #generate spectral_contrast_stft features
    filedata['spectral_contrast_stft'] = Parallel(n_jobs=-1)(delayed(sonicboom.spectral_contrast_stftEngineering)(x) for x in filedata['path'])
    print("Spectral contrast (STFT) done!")

if (tonnetz_exec == True):
    #generate tonnetz features
    filedata['tonnetz'] = Parallel(n_jobs=-1)(delayed(sonicboom.tonnetzEngineering)(x) for x in filedata['path'])
    print("Tonal centroid features (tonnetz) done!")

endTime = time.perf_counter()
runTime = endTime - startTime
print(f'Finished in {runTime:.4f} secs')

filedata.head()

# %% [markdown]
# ## Save the generated features
filedata.to_pickle(SAVEPATH + FILEDESC)

# %%
