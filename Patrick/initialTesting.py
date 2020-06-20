# %% Imports

#general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import os
import pathlib

#audio processing/tools import
import librosa
import librosa.display
from scipy.io.wavfile import read 
from IPython.display import Audio
#REMEMBER you need ffmpeg installed

#other imports
import glob #filesystem manipulation

# %% Read the metadata
metaData = pd.read_csv('../.data/UrbanSound8K/metadata/UrbanSound8K.csv')


# %% Generate file paths and merge with metadata

#recursively add all .wave files to paths list
paths = list(pathlib.Path('../.data/UrbanSound8K/audio/').glob('**/*.wav'))

filenames = paths.copy()

#remove path from filenames leaving us just with the raw filename
for i in range(len(filenames)):
      filenames[i] = os.path.basename(filenames[i])

#create dataframe from paths and filenames
filedata = pd.DataFrame(list(zip(paths, filenames)), columns =['path', 'slice_file_name']) 

#merge metadata and filedata (the one we just created) dataframes
filedata = filedata.join(metaData.set_index('slice_file_name'), on='slice_file_name')

# %% sample down

#samples down grouping by class - this gives me 15 (or whatever number) items from each class.
#as_index=False is important because otherwise Pandas calls the index and the column the same thing, confusing itself
filedata = filedata.groupby('class', as_index=False).apply(lambda x: x.sample(100))

#check that the sample down is working
#as_index=False is important because otherwise Pandas calls the index and the column the same thing, confusing itself
filedata.groupby('class', as_index=False)['slice_file_name'].nunique()

# %% loop through and do things to individual audio files (generate features)



#audioFile, sampling_rate = librosa.load('../.data/UrbanSound8K/audio/patrickTemp/19026-1-0-0.wav')

audioFile, sampling_rate = librosa.load('../.data/UrbanSound8K/audio/fold1/103074-7-0-0.wav')

#sf.write('testFile.wav', audioFile, sampling_rate, subtype='PCM_16')



# %% loop through and do things to individual audio files (generate features)

def mfccsEngineering(filepath):
      audioFile, sampling_rate = librosa.load(filepath)
      mfccs = librosa.feature.mfcc(y=audioFile, sr=sampling_rate,  n_mfcc=40)
      mfccs = np.mean(mfccs.T,axis=0)
      return mfccs

filedata['mfccs'] = [mfccsEngineering(x) for x in filedata['path']]

filedata.head()

filedata.to_csv('./mfccsFeature.csv')

# %% Plot librosa audio visualizations


# %% Plot librosa audio visualizations
plt.figure(figsize=(12, 4))
librosa.display.waveplot(audioFile, sr=sampling_rate)

#plt.figure(figsize=(12, 4))
#librosa.display.specshow(audioFile)

# %% Feature Engineering with Librosa

#Mel-frequency cepstral coefficients (MFCCs)

mfccs = librosa.feature.mfcc(y=audioFile, sr=sampling_rate,  n_mfcc=40)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()


# %% quick and dirty model exec - logistic regression

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import label_binarize

y = label_binarize(filedata['classID'], classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

X = list((filedata['mfccs']))
#y = list(filedata['classID'])

logregmodel = OneVsRestClassifier(LogisticRegression(random_state=1)).fit(X, y)


# %% score function - log reg

filedata['predictedClass'] = logregmodel.predict(list(filedata['mfccs']))

filedata.to_csv('./binarize_test.csv')

#accuracy = logregmodel.score(X=list((filedata['mfccs'])), y=list(filedata['classID']))
#accuracy = accuracy * 100

#print ("The accuracy of this classifier is", accuracy, "%") 

#filedata.head()

# %% quick and dirty model exec - SVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

X = list((filedata['mfccs']))
y = list(filedata['classID'])
svcmodel = OneVsRestClassifier(SVC(random_state=1)).fit(X, y)


# %% score function - SVC

filedata['predictedClass'] = svcmodel.predict(list(filedata['mfccs']))

accuracy = svcmodel.score(X=list((filedata['mfccs'])), y=list(filedata['classID']))

accuracy = accuracy * 100

print ("The accuracy of this classifier is", accuracy, "%") 

#filedata.head()

#filedata.to_csv('./DONE.csv')


# %% quick and dirty model exec - KNN

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

X = list((filedata['mfccs']))
y = list(filedata['classID'])
KNNmodel = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X, y)


# %% score function - KNN

filedata['predictedClass'] = KNNmodel.predict(list(filedata['mfccs']))

accuracy = KNNmodel.score(X=list((filedata['mfccs'])), y=list(filedata['classID']))

accuracy = accuracy * 100

print ("The accuracy of this classifier is", accuracy, "%") 

#filedata.head()

#filedata.to_csv('./KNN.csv')

# %% all of em 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
        GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier

models = [  
    OneVsRestClassifier(LogisticRegression(random_state=1)),
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)),
    OneVsRestClassifier(DecisionTreeClassifier()),
    OneVsRestClassifier(GaussianNB())
]

X = list((filedata['mfccs']))
y = list(filedata['classID'])

for model in models:
      modeloutput = (model.fit(X, y))
      accuracy = model.score(X=list((filedata['mfccs'])), y=list(filedata['classID']))
      accuracy = accuracy * 100
      print ("The accuracy of this classifier is", accuracy, "%") 


# %%
