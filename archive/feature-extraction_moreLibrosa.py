# %% Imports

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

# Import custom module containing useful functions
import sonicboom

# Define some decorator functions
import time

# %% Read the metadata

filedata = sonicboom.init_data('./data/UrbanSound8K/')

# %% Sample down

# samples down grouping by class - this gives me 15 (or whatever number) items from each class.
# as_index=False is important because otherwise Pandas calls the index and the column the same thing, confusing itself
filedata = filedata.groupby('class', as_index=False).apply(lambda x: x.sample(2))

# check that the sample down is working
# as_index=False is important because otherwise Pandas calls the index and the column the same thing, confusing itself
filedata.groupby('class', as_index=False)['slice_file_name'].nunique()

# %% Read one audio file to see what it contains

sonicboom.test_read_audio(filedata.path.iloc[16])


# %% PARALLEL Generate MFCCs and add to dataframe
from joblib import Parallel, delayed

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
spectral_contrast_exec = True
tonnetz_exec = True

# if (inputVar == 0):
#     mfccs_exec = True
#     melSpec_exec = True
#     stft_exec = True
#     chroma_stft_exec = True
#     spectral_contrast_exec = True
#     tonnetz_exec = True
# elif (inputVar == 1):
#     mfccs_exec = True
# elif (inputVar == 2):
#     melSpec_exec = True
# elif (inputVar == 3):
#     stft_exec = True
# elif (inputVar == 4):
#     chroma_stft_exec = True
# elif (inputVar == 5):
#     spectral_contrast_exec = True
# elif (inputVar == 6):
#     tonnetz_exec = True

if (mfccs_exec == True):
    #generate mfccs features
    filedata['mfccs'] = Parallel(n_jobs=-1)(delayed(sonicboom.mfccsEngineering)(x) for x in filedata['path'])

if (melSpec_exec == True):
    #generate melSpec features
    filedata['melSpec'] = Parallel(n_jobs=-1)(delayed(sonicboom.melSpecEngineering)(x) for x in filedata['path'])

if (stft_exec == True):
    #generate stft features
    filedata['stft'] = Parallel(n_jobs=-1)(delayed(sonicboom.stftEngineering)(x) for x in filedata['path'])

if (chroma_stft_exec == True):
    #generate chroma_stft features
    filedata['chroma_stft'] = Parallel(n_jobs=-1)(delayed(sonicboom.chroma_stftEngineering)(x) for x in filedata['path'])

if (spectral_contrast_exec == True):
    #generate spectral_contrast features
    filedata['spectral_contrast'] = Parallel(n_jobs=-1)(delayed(sonicboom.spectral_contrastEngineering)(x) for x in filedata['path'])

if (tonnetz_exec == True):
    #generate tonnetz features
    filedata['tonnetz'] = Parallel(n_jobs=-1)(delayed(sonicboom.tonnetzEngineering)(x) for x in filedata['path'])

endTime = time.perf_counter()
runTime = endTime - startTime
print(f'Finished in {runTime:.4f} secs')

filedata.head()

#filedata.to_csv('./mfccsFeature.csv')


# %% Parallel check

#for x in range(len(filedata)):
#    print(np.array_equal(filedata['mfccs'].iloc[x], filedata['mfccsParallel'].iloc[x]))

# %% prep features for models

#take mean of transposed mfccs (for some reason?) - this is now done in SonicBoom
#filedata['mfccs'] = [np.mean(x.T,axis=0) for x in filedata['mfccs']]

# %% Initial model generation: all of em 
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

# %%
