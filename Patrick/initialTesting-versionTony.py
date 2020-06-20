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

#audioFile, sampling_rate = librosa.load('../.data/UrbanSound8K/audio/fold1/103074-7-0-0.wav')

def mfccsEngineering(filepath):
      audioFile, sampling_rate = librosa.load(filepath)
      mfccs = librosa.feature.mfcc(y=audioFile, sr=sampling_rate,  n_mfcc=40)
      mfccs = np.mean(mfccs.T,axis=0)
      return mfccs

filedata['mfccs'] = [mfccsEngineering(x) for x in filedata['path']]

filedata.head()

#filedata.to_csv('./mfccsFeature.csv')

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
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from mlxtend.plotting import plot_learning_curves

models = [  
    OneVsRestClassifier(LogisticRegression(random_state=1)),
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)),
    OneVsRestClassifier(DecisionTreeClassifier()),
    OneVsRestClassifier(GaussianNB()),
    OneVsRestClassifier(LinearSVC())
]
model_namelist = ['Logistic Regression',
                  'KNeighbors',
                  'Decision Tree',
                  'GaussianNB', 
                  'SVM/Linear SVC'
                  ]
scoring = {'precision': make_scorer(precision_score, average='micro'), 
           'recall': make_scorer(recall_score, average='micro'), 
           'f1': make_scorer(f1_score, average='micro'),
           'roc_auc': make_scorer(roc_auc_score, average='micro'),
           # 'mcc': make_scorer(matthews_corrcoef) <- cannot support multi-label
          }

cv_result_entries = []
i = 0                  

X = pd.DataFrame(filedata['mfccs'].iloc[x] for x in range(len(filedata)))
y = label_binarize(
      pd.DataFrame(filedata['classID'].iloc[x] for x in range(len(filedata))),
      classes=[0,1,2,3,4,5,6,7,8,9]
      )

# ### Loop cross validation through various models and generate results
for mod in models:
    metrics = cross_validate(
        mod,
        X,
        y,
        cv=5,
        scoring = scoring,
        return_train_score=False,
        n_jobs=-1
    )
    for key in metrics.keys():
        for fold_index, score in enumerate(metrics[key]):
            cv_result_entries.append((model_namelist[i], fold_index, key, score))
    i += 1
cv_results_df = pd.DataFrame(cv_result_entries)

# %%
# ### Misclassification Errors
i=0
for model in models:
    plt.figure()
    plot_learning_curves(X, y, X, y, model)
    plt.title('Learning Curve for ' + model_namelist[i], fontsize=14)
    plt.xlabel('Training Set Size (%)', fontsize=12)
    plt.ylabel('Misclassification Error', fontsize=12)
    plt.show()
    i += 1

# %% [markdown]
# ### Get predictions: prep for Confusion Matrix
y_test_pred = []
for model in models:
    y_test_pred.append(model.predict(X))

# %% [markdown]
# ### Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASSES = ['A/C', 'Car Horn', 'Children Play', 'Dog Bark',
           'Drilling', 'Engine Idle', 'Gun Shot', 'Jackhammer',
           'Siren', 'Street Music']
i=0
for _ in models:
    cm = confusion_matrix(np.argmax(y, axis=1),
                          np.argmax(y_test_pred[i], axis=1))
    cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.title('Confusion Matrix for ' + model_namelist[i], fontsize=14)
    sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
    plt.show()
    i += 1