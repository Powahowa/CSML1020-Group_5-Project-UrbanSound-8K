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
# Image classification using Tensorflow & Keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import itertools

#%%
# Set folder paths
src = '../CSML1020-Group_5-Project-UrbanSound-8K/output'
train_dir = src + '/train' # training directory
validation_dir = src + '/validation' # validation directory

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
#For convenience, set up variables to use 
#while pre-processing the dataset and training the network
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# %%
# Data Generator
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode= 'categorical' )

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

#%% 
# Visualize Training Images
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#%% 
# Plot images
plotImages(sample_training_images[:10])

# %%
""" # Visualize Training Images
train_folders = [train_air_conditioner_dir, train_car_horn_dir,train_children_playing_dir
                ,train_dog_bark_dir, train_drilling_dir, train_engine_idling_dir,
                train_gun_shot_dir, train_jackhammer_dir, train_siren_dir,
                train_street_music_dir]

img = []

#%%
for r in range(10):
    
    for i in range(10):
        imag = os.listdir(train_folders[r])[i]
        img.append(imag)

#%% 
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for i, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show() """

#%% 
# Create the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

#%%
# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# model summary
model.summary()

# %%
# Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#%%
# Visualize Training Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#%%
# Data Augmentation - Horizontal Flip
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(10)]

# Re-use the same custom plotting function defined and used
# above to visualize the training images
plotImages(augmented_images)

#%% 
# Data Augmentation - Rotation
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)  

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(10)]

plotImages(augmented_images)

#%%
# Data Augmentation - Zoom
# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) # 

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(10)]

plotImages(augmented_images)

#%%
# Put it all together
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

augmented_images = [train_data_gen[0][0][0] for i in range(10)]
plotImages(augmented_images)

#%%
# Create Validation data generator
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')

#%%
# New network with Dropouts
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

#%%
# Compile the model
model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()

#%%
# Train the model
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

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
