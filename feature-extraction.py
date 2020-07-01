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
MFCCFILENAME = 'filedata-mfcc.csv'

# %% [markdown]
# ## Read and add filepaths to original UrbanSound metadata
filedata = sonicboom.init_data('./data/UrbanSound8K/')

# %% [markdown]
# ## Sample down

# samples down grouping by class - this gives me X items from each class.
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
filedata = filedata.groupby('class', as_index=False).apply(lambda x: x.sample(100))

# check that the sample down is working
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
filedata.groupby('class', as_index=False)['slice_file_name'].nunique()

# %% [markdown]
# ## Read one audio file to see what it contains
sonicboom.test_read_audio(filedata.path.iloc[0])

# %% [markdown]
# ## PARALLEL Generate MFCCs and add to dataframe
startTime = time.perf_counter()

filedata['mfccs'] = Parallel(n_jobs=-1)(delayed(
    sonicboom.mfccsEngineering)(x) for x in filedata['path'])

# non-parallel version
# filedata['mfccs'] = [sonicboom.mfccsEngineering(x) for x in filedata['path']]

endTime = time.perf_counter()
runTime = endTime - startTime
print(f'Finished in {runTime:.4f} secs')

filedata.head()

# %% [markdown]
# ## Save the generated features
filedata.to_csv(SAVEPATH + MFCCFILENAME, index=False)