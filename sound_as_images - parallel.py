# %% [markdown]

# This python script takes audio files from "filedata" from sonicboom, runs each audio file through 
# Fast Fourier Transform, plots the FFT image, splits the FFT'd images into train, test & validation
# and paste them in their respective folders

# Import Dependencies

import numpy as np
import pandas as pd

import scipy 
from scipy import io
from scipy.io.wavfile import read as wavread
from scipy.fftpack import fft

import librosa
from librosa import display

import matplotlib.pyplot as plt 

from glob import glob

import sklearn
from sklearn.model_selection import train_test_split

import os

from PIL import Image

import pathlib

import sonicboom

from joblib import Parallel, delayed
# %% [markdown]
# ## Read and add filepaths to original UrbanSound metadata
filedata = sonicboom.init_data('./data/UrbanSound8K/') #Read filedata as written in sonicboom

#Initialize empty dataframes to later enable saving the images into their respective folders
train = pd.DataFrame()
test = pd.DataFrame()
validation = pd.DataFrame()

""" for x = in range(10):
    fileclass = filedata['classID'] == x
    filtered = filedata[fileclass]
    trainTemp, testTemp = train_test_split(filtered, test_size=0.20, random_state=0)
    train = pd.concat([train, trainTemp])
    test = pd.concat([test, testTemp]) """
# %%
# Read the entire filedata
filtered = filedata #Read the data in 
trainTemp, valTemp = train_test_split(filtered, test_size=0.10, random_state=0) #Take 10% as blind test set
trainTemp, testTemp = train_test_split(trainTemp, test_size=0.20, random_state=0) #Split the remaining into an 80-20 train test split

# Assign the different splits to different dataframes
train = pd.concat([train, trainTemp]) 
test = pd.concat([test, testTemp]) 
validation = pd.concat([validation, valTemp])

#%%
#Read all files in each path folder iteration

def fftgen(filepath, filename, classID, split):

        audio, sfreq = librosa.load(filepath)        
        
        #FFT
        ftrans = abs(np.fft.fft(audio, n=88200)) #[:round((audio.size/2))])
        ftrans_pos = ftrans[:round(ftrans.size/2)]
        #fr = numpy.fft.fftfreq(len(ftrans))

        # Steps to filter > 0 values in fr
        #fr = fr[fr >= 0]
        
        # Plot the FFT
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        
        ax = plt.Axes(fig,[0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        fig = plt.plot(ftrans_pos)

        if split == 'train':
            fname = filename
            folder = classID
            img_path = 'output/train/' + folder + '/' + fname + '.png'
        elif split == 'test':
            fname = filename
            folder = classID
            img_path = 'output/test/' + folder + '/' + fname + '.png'
        elif split == 'validation':
            fname = filename
            folder = classID
            img_path = 'output/validation/' + folder + '/' + fname + '.png' 

        plt.savefig(img_path, dpi = 25.6) 
        plt.close()    

#%% Parallel
Parallel(n_jobs=-1)(delayed(fftgen) \
    (filepath = test['path'].iloc[j], filename= test['slice_file_name'].iloc[j], classID=test['class'].iloc[j], split = "test") for j in range(len(test)))

#%%
Parallel(n_jobs=-1)(delayed(fftgen) \
    (filepath = train['path'].iloc[j], filename= train['slice_file_name'].iloc[j], classID=train['class'].iloc[j], split = "train") for j in range(len(train)))

#%%

Parallel(n_jobs=-1)(delayed(fftgen) \
    (filepath = validation['path'].iloc[j], filename= validation['slice_file_name'].iloc[j], classID=validation['class'].iloc[j], split = "validation") for j in range(len(validation)))









# %% [markdown]
# Code References
#* https://www.youtube.com/watch?v=vJ_WL9aYfNI
#* https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/#:~:text=Glob%20is%20a%20general%20term,pathnames%20matching%20a%20specified%20pattern.
#* https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame
#* https://www.geeksforgeeks.org/working-images-python/ (working with Pillow)
#* https://stackoverflow.com/questions/58089062/logistic-regression-in-python-with-a-dataframe-of-arrays


# References
#* https://docs.python.org/3/library/wave.html
#* https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html
#* https://www.youtube.com/watch?v=0ALKGR0I5MA
#* https://www.youtube.com/watch?v=Z7YM-HAz-IY&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P
#* https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
#* https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
#* https://www.youtube.com/watch?v=aQKX3mrDFoY
#* https://heartbeat.fritz.ai/build-a-deep-learning-model-to-classify-images-using-keras-and-tensorflow-2-0-379e99c0ba88
#* https://www.tensorflow.org/tutorials/images/classification
#* https://www.kaggle.com/theimgclist/multiclass-food-classification-using-tensorflow
#* https://www.kaggle.com/theimgclist/multiclass-food-classification-using-tensorflow
# %% 
# Shady References
#* https://www.youtube.com/watch?v=17cOaqrwXlo
#* https://www.dummies.com/programming/python/performing-a-fast-fourier-transform-fft-on-a-sound-file/
#* https://www.youtube.com/watch?v=17cOaqrwXlo


# %%
