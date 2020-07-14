# %% [markdown]
# Import Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
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
filedata = sonicboom.init_data('./data/UrbanSound8K/')

train = pd.DataFrame()
test = pd.DataFrame()

""" for x = in range(10):
    fileclass = filedata['classID'] == x
    filtered = filedata[fileclass]
    trainTemp, testTemp = train_test_split(filtered, test_size=0.20, random_state=0)
    train = pd.concat([train, trainTemp])
    test = pd.concat([test, testTemp]) """
# %%
fileclass = filedata['classID'] == 9
filtered = filedata[fileclass]
trainTemp, testTemp = train_test_split(filtered, test_size=0.20, random_state=0)
train = pd.concat([train, trainTemp])
test = pd.concat([test, testTemp])


#%%
#Read all files in each path folder iteration

def fftgen(filepath, filename, classID, split):

        audio, sfreq = librosa.load(filepath)        
        
        #FFT
        ftrans = abs(numpy.fft.fft(audio, n=88200)) #[:round((audio.size/2))])
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
            img_path = 'output/validation/' + folder + '/' + fname + '.png'

        plt.savefig(img_path, dpi = 25.6) 
        plt.close()    

#%%
fftgen(test, split = "test")
#%%
fftgen(train, split = "train")

#%%
for j in range(len(test)):
    fftgen(filepath = test['path'].iloc[j], filename= test['slice_file_name'].iloc[j], classID=test['class'].iloc[j], split = "test")

#%% Parallel
Parallel(n_jobs=-1)(delayed(fftgen) \
    (filepath = test['path'].iloc[j], filename= test['slice_file_name'].iloc[j], classID=test['class'].iloc[j], split = "test") for j in range(len(test)))

#%%

Parallel(n_jobs=-1)(delayed(fftgen) \
    (filepath = train['path'].iloc[j], filename= train['slice_file_name'].iloc[j], classID=train['class'].iloc[j], split = "train") for j in range(len(train)))



#%% 
# Multiclass with INCEPTION
#import dependencies
import tensorflow as tf
import matplotlib.image as img
%matplotlib inline
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2
#%%
# Use the pretrained INCEPTION model
K.clear_session()

n_classes = 10
#img_width, img_height = 299, 299
train_data_dir = train_dir
validation_data_dir = validation_dir
nb_train_samples = 6985
nb_validation_samples = 1747
batch_size = 16
#%%
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=batch_size,
    class_mode='categorical')
#%%
inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n_classes,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_11class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_11class.log')

sound_10class = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=5,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_sound_10.hdf5')









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
