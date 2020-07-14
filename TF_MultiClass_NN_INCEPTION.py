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

#%%
# Set folder paths
src = '../CSML1020-Group_5-Project-UrbanSound-8K/output'
train_dir = src + '/train' # training directory
validation_dir = src + '/test' # validation directory (In this script, "validation" actually means "test")

train_air_conditioner_dir = train_dir + '/air_conditioner' #directory with training AC fft plots
train_car_horn_dir = train_dir + '/car_horn' #directory with training carhorn  fft plots
train_children_playing_dir = train_dir + '/children_playing' #directory with training children playing fft plots
train_dog_bark_dir = train_dir + '/dog_bark' #directory with training dogbark fft plots
train_drilling_dir = train_dir + '/drilling' #directory with training drilling fft plots
train_engine_idling_dir = train_dir + '/engine_idling' #directory with training engine idling fft plots
train_gun_shot_dir = train_dir + '/gun_shot' #directory with training gun shot fft plots
train_jackhammer_dir = train_dir + '/jackhammer' #directory with training jackhammer fft plots
train_siren_dir = train_dir + '/siren' #directory with training siren fft plots
train_street_music_dir = train_dir + '/street_music' #directory with training street music fft plots

validation_air_conditioner_dir = validation_dir + '/air_conditioner' #directory with validation AC fft plots
validation_car_horn_dir = validation_dir + '/car_horn' #directory with validation carhorn  fft plots
validation_children_playing_dir = validation_dir + '/children_playing' #directory with validation children playing fft plots
validation_dog_bark_dir = validation_dir + '/dog_bark' #directory with validation dogbark fft plots
validation_drilling_dir = validation_dir + '/drilling' #directory with validation drilling fft plots
validation_engine_idling_dir = validation_dir + '/engine_idling' #directory with validation engine idling fft plots
validation_gun_shot_dir = validation_dir + '/gun_shot' #directory with validation gun shot fft plots
validation_jackhammer_dir = validation_dir + '/jackhammer' #directory with validation jackhammer fft plots
validation_siren_dir = validation_dir + '/siren' #directory with validation siren fft plots
validation_street_music_dir = validation_dir + '/street_music' #directory with validation street music fft plots

#%%
# Understand the data
num_air_conditioner_tr = len(os.listdir(train_air_conditioner_dir))
num_car_horn_tr = len(os.listdir(train_car_horn_dir))
num_children_playing_tr = len(os.listdir(train_children_playing_dir))
num_dog_bark_tr = len(os.listdir(train_dog_bark_dir))
num_drilling_tr = len(os.listdir(train_drilling_dir))
num_engine_idling_tr = len(os.listdir(train_engine_idling_dir))
num_gun_shot_tr = len(os.listdir(train_gun_shot_dir))
num_jackhammer_tr = len(os.listdir(train_jackhammer_dir))
num_siren_tr = len(os.listdir(train_siren_dir))
num_street_music_tr = len(os.listdir(train_street_music_dir))

num_air_conditioner_val = len(os.listdir(validation_air_conditioner_dir))
num_car_horn_val = len(os.listdir(validation_car_horn_dir))
num_children_playing_val = len(os.listdir(validation_children_playing_dir))
num_dog_bark_val = len(os.listdir(validation_dog_bark_dir))
num_drilling_val = len(os.listdir(validation_drilling_dir))
num_engine_idling_val = len(os.listdir(validation_engine_idling_dir))
num_gun_shot_val = len(os.listdir(validation_gun_shot_dir))
num_jackhammer_val = len(os.listdir(validation_jackhammer_dir))
num_siren_val = len(os.listdir(validation_siren_dir))
num_street_music_val = len(os.listdir(validation_street_music_dir))

total_train = (num_air_conditioner_tr + num_car_horn_tr + num_children_playing_tr +
            num_dog_bark_tr + num_drilling_tr + num_engine_idling_tr + num_gun_shot_tr+
            num_jackhammer_tr + num_siren_tr + num_street_music_tr)

total_val = (num_air_conditioner_val + num_car_horn_val + num_children_playing_val +
            num_dog_bark_val + num_drilling_val + num_engine_idling_val + num_gun_shot_val+
            num_jackhammer_val + num_siren_val + num_street_music_val)

print('total training air_conditioner images:', num_air_conditioner_tr)
print('total training car_horn images:', num_car_horn_tr)
print('total training children_playing images:', num_children_playing_tr)
print('total training dog_bark images:', num_dog_bark_tr)
print('total training drilling images:', num_drilling_tr)
print('total training engine_idling images:', num_engine_idling_tr)
print('total training gun_shot images:', num_gun_shot_tr)
print('total training jackhammer images:', num_jackhammer_tr)
print('total training siren images:', num_siren_tr)
print('total training street_music images:', num_street_music_tr)

print('total validation air_conditioner images:', num_air_conditioner_val)
print('total validation car_horn images:', num_car_horn_val)
print('total validation children_playing images:', num_children_playing_val)
print('total validation dog_bark images:', num_dog_bark_val)
print('total validation drilling images:', num_drilling_val)
print('total validation engine_idling images:', num_engine_idling_val)
print('total validation gun_shot images:', num_gun_shot_val)
print('total validation jackhammer images:', num_jackhammer_val)
print('total validation siren images:', num_siren_val)
print('total validation street_music images:', num_street_music_val)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

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
nb_train_samples = 6286
nb_validation_samples = 1572
batch_size = 20
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
checkpointer = ModelCheckpoint(filepath='sound_best_model_10class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_10class.log')
#%%
sound_10class = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=100,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_sound_10.hdf5')

#%%
class_map_10 = train_generator.class_indices
class_map_10

#%%
def plot_accuracy(history,title):
    plt.title(title) 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()
def plot_loss(history,title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()

#%%
plot_accuracy(sound_10class,'Sound-Inceptionv3')
plot_loss(sound_10class,'Sound-Inceptionv3')

#%%

# Loading the best saved model to make predictions
K.clear_session()
model_best = load_model('sound_best_model_10class.hdf5',compile = False)

#%% Set blind test directory
blind_dir = src + '/validation' # blind test directory

#%% Load blind test images
print("This is the blind test")
print('------------------------')

images = []

for n in range(1):# len(os.listdir(blind_dir))):
    images.append(blind_dir + '/' + os.listdir(blind_dir)[n])

#%%
def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img) #target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    class_list.sort()
    pred_value = class_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()

#%% 
predict_class(model_best, images, True)

#%%

#%%

# %% [markdown]
# Code References
#* https://www.youtube.com/watch?v=vJ_WL9aYfNI
#* https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/#:~:text=Glob%20is%20a%20general%20term,pathnames%20matching%20a%20specified%20pattern.
#* https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame
#* https://www.geeksforgeeks.org/working-images-python/ (working with Pillow)
#* https://stackoverflow.com/questions/58089062/logistic-regression-in-python-with-a-dataframe-of-arrays

#%%
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
