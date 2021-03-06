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
#FILEDESC = 'filedata-librosaFeatures.pkl'
FILEDESC = 'filedata-librosaConventionalFeatures.pkl'

# %% [markdown]
# ## Read and add filepaths to original UrbanSound metadata
filedata = sonicboom.init_data('./data/UrbanSound8K/')

# %% [markdown]
# ## Sample down

sampleDown = True

# samples down grouping by class - this gives me X items from each class.
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
if (sampleDown == True):
    filedata = filedata.groupby(
        'class', 
        as_index=False, 
        group_keys=False
    ).apply(lambda x: x.sample(n=3, random_state=0))

# check that the sample down is working
# as_index=False is important because otherwise,
# Pandas calls the index and the column the same thing, confusing itself
filedata.groupby('class', as_index=False)['slice_file_name'].nunique()


# %%

#Feature plot generation

Parallel(n_jobs=-1)(delayed(sonicboom.featurePlot)(filedata['path'].iloc[x], \
filedata['slice_file_name'].iloc[x], filedata['classID'].iloc[x]) for x in range(len(filedata)))

#for x in range(len(filedata)):
#    sonicboom.featurePlot()




# %% [markdown]
# ## Read one audio file to see what it contains
sonicboom.test_read_audio(filedata.path.iloc[0])

# %%
# Get sample rates for all .wave files and add to filedata dataframe

#filedata['Sample Rate'] = Parallel(n_jobs=-1)(delayed(sonicboom.samplerate)(x) for x in filedata['path'])

#filedata.to_csv('./SampleRates.csv')

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
#     7. Vis's custom FFT feature

mfccs_exec = True
melSpec_exec = True
stft_exec = False #too many elements, array is huge, cannot be flattened
chroma_stft_exec = True 
spectral_contrast_stft_exec = True
tonnetz_exec = True
visFFT_exec = False #huge, cannot be flattened
mfccDelta_exec = False #for neural network only, cannot be normalized
flatten = True
normalize = True

tempDF = pd.DataFrame() 

tempDF = pd.concat(Parallel(n_jobs=-1)(delayed(sonicboom.generateFeatures) \
    (x, mfccs_exec, melSpec_exec, stft_exec, chroma_stft_exec, \
        spectral_contrast_stft_exec, tonnetz_exec, visFFT_exec, mfccDelta_exec, \
        flatten, normalize) for x in filedata['path']))

filedata = filedata.join(tempDF.set_index('path'), on='path')

endTime = time.perf_counter()
runTime = endTime - startTime
print(f'Finished in {runTime:.4f} secs')

filedata.head()

# %% [markdown]
# ## Save the generated features
filedata.to_pickle(SAVEPATH + FILEDESC)
# %% 

#working cell for plotting

audioFile, samplingRate = sonicboom.load_audio(filedata['path'].iloc[0])

plt.figure(figsize=(10, 4))

tonnetz = sonicboom.tonnetzEngineering(audioFile, samplingRate)

plt.figure(figsize=(8, 4))
librosa.display.specshow(tonnetz, y_axis='tonnetz')
plt.colorbar()
plt.title('Tonal Centroids (Tonnetz)')

# %%
