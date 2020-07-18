# %% [markdown]
# Import Dependencies
import numpy
import pandas
from pandas import merge
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


# %% [markdown]
# Read in the meta-data file
meta = pandas.read_csv('../data/UrbanSound8K/metadata/UrbanSound8K.csv')
meta_fold1 = meta[meta.fold == 1]

# %% [markdown]
""" The idea is to read all sound files found in the meta data list, put them through a fourier
transform and save the images of the transformed signal to perform classification on """

#%% 
# # Set directory to read files from 
src = '../data/UrbanSound8K/audio'

# Recursively read files
# Get a dataframe of all folder names
# fold_df = pandas.DataFrame(next(os.walk(src))[1])



#%%
# Initiate lists to hold all the raw file, it's timeline, fft'd audio files, filename
raw_list = []
timeline = []
fft_file = []
fft_freq = []
filename = []
temp_df = pandas.DataFrame()

# Loop through each folder & read all files
""" for i in range(0,len(fold_df)):
    fold = fold_df.iloc[i,0]
    path = '{}/{}'.format(src, ''.join(fold)) """

path = src + '/fold1'

#%%
#Read all files in each path folder iteration
files = glob(path + '/*.wav')
for j in range(0,10):#(len(files))):
    audio, sfreq = librosa.load(files[j])        
    raw_list.append(audio)
    filename.append(os.path.basename(files[j]))
    
    #FFT
    ftrans = abs(numpy.fft.fft(audio, n=88200)) #[:round((audio.size/2))])
    ftrans_pos = ftrans[:round(ftrans.size/2)]
    fr = numpy.fft.fftfreq(len(ftrans))

    # Steps to filter > 0 values in fr
    filter = [] #An empty list for filtering

    fr = fr[fr >= 0]
    fr = fr.ravel()

    fft_freq.append(fr)
    
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    fig = plt.plot(ftrans_pos)

    #ax.imshow(fig, aspect='auto')
    #plt.savefig(f'test{j}', dpi = 25.6)
    plt.savefig(os.path.splitext(os.path.basename(files[j]))[0], dpi = 25.6)
    plt.close()

    temp_df = temp_df.append({'fft_freq_array': fft_freq}, ignore_index=True)
# %%
#temp_df = temp_df.append({'slice_file_name': filename}, ignore_index=True)    
temp_df['slice_file_name'] = filename
# %% 
sound_df = meta_fold1.merge(temp_df, on = 'slice_file_name', how = 'left')

# %%
# Get x & y from "sound_df"
x = sound_df.iloc[:, sound_df.columns == 'fft_freq_array']
y = sound_df.iloc[:, sound_df.columns == 'classID']

#%% 
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=9)

""" x_train = x.head(round(0.7*len(x)))
x_test = x.tail(round(0.3*len(x)))

y_train = y.head(round(0.7*len(y)))
y_test = y.tail(round(0.7*len(y))) """


# %% 
## Using FFT Freqs as features
# Basic classification models 
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

""" # %% 
# Define models
def run_logreg(x_train, y_train):
    classifier = OneVsRestClassifier(LogisticRegression(random_state=9))
    classifier.fit(x_train, y_train)
    return classifier

def knn(x_train, y_train):
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
    classifier.fit(x_train, y_train)
    return classifier

def run_dectree(x_train, y_train):
    classifier = OneVsRestClassifier(DecisionTreeClassifier())
    classifier.fit(x_train, y_train)
    return classifier 

def run_nb(x_train, y_train):
    classifier = OneVsRestClassifier(GaussianNB())
    classifier.fit(x_train, y_train)
    return classifier

def run_svm(x_train, y_train):
    classifier = OneVsRestClassifier(LinearSVC(random_state=9))
    classifier.fit(x_train, y_train)
    return classifier """

#%%
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

x_train_val = x_train.fft_freq_array.ravel()

for m in models:
    classifier = m
    classifier.fit(x_train,y_train)
    #return classifier

cv_result_entries = []
i = 0                  

#X = pandas.DataFrame(sound_df['fft_freq_array'].iloc[x] for x in range(len(sound_df)))
""" y = label_binarize(
      pandas.DataFrame(sound_df['classID'].iloc[x] for x in range(len(sound_df))),
      classes=[0,1,2,3,4,5,6,7,8,9]
      )
 """
# ### Loop cross validation through various models and generate results
for mod in models:
    metrics = cross_validate(
        mod,
        x,
        y,
        cv=5,
        scoring = scoring,
        return_train_score=False,
        n_jobs=2
    )
    for key in metrics.keys():
        for fold_index, score in enumerate(metrics[key]):
            cv_result_entries.append((model_namelist[i], fold_index, key, score))
    i += 1
cv_results_df = pandas.DataFrame(cv_result_entries)

# %% [markdown]
# ### Misclassification Errors
i=0
for model in models:
    plot_learning_curves(x_train_val, y_train, x_test, y_test, model)
    plt.title('Learning Curve for ' + model_namelist[i], fontsize=14)
    plt.xlabel('Training Set Size (%)', fontsize=12)
    plt.ylabel('Misclassification Error', fontsize=12)
    plt.show()
    i += 1

# %% [markdown]
# ### Get predictions: prep for Confusion Matrix
y_test_pred = []
for model in models:
    y_test_pred.append(model.predict(x_test))

# %% [markdown]
# ### Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASSES = ['A/C', 'Car Horn', 'Children Play', 'Dog Bark',
           'Drilling', 'Engine Idle', 'Gun Shot', 'Jackhammer',
           'Siren', 'Street Music']
i=0
for _ in models:
    cm = confusion_matrix(numpy.argmax(y, axis=1),
                          numpy.argmax(y_test_pred[i], axis=1))
    cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.title('Confusion Matrix for ' + model_namelist[i], fontsize=14)
    sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
    plt.show()
    i += 1

















# %% 
# Plotting waves on time domain
""" plt.figure()
librosa.display.waveplot(y = raw_list[872], sr = sfreq)
plt.xlabel("Time in secs")
plt.ylabel("Amplitude")
plt.show() """


""" # %% [markdown]
## Data Exploration
# View data by class 
class_count = meta['class'].value_counts() """

 













# %% [markdown]
# Read in Data using scipy (sample)

""" testfile = scipy.io.wavfile.read("./data/UrbanSound8K/audio/fold1/7383-3-0-1.wav")
aud_array = testfile[1]
if aud_array.ndim == 2: 
    aud_array = numpy.mean(testfile[1], axis = 1) #averages the two channels of a stereo signal 
else:
    aud_array = testfile[1]
 """

""" #%% 
# Read in data using Librosa
audio, sfreq = librosa.load(sound_file[4])

# %%
timeline = numpy.arange(0, len(audio)) / sfreq """

# %% 
# Manually plot on "time domain"
""" time = numpy.linspace(0, len(aud_array)/testfile[0], num=len(aud_array))
plt.plot(aud_array)
plt.show()
plt.plot(time, aud_array)
plt.show() """

""" # %% 
# Plot audio over time using Librosa inputs
fig, ax = plt.subplots()
ax.plot(timeline, audio)
ax.set(xlabel = "Time in secs", ylabel = "Amplitude")
plt.show() """

# %% 
""" # Fourier Transform
ftrans = numpy.fft.fft(audio) #[:round((audio.size/2))])
ftrans_pos = ftrans[:round(ftrans.size/2)]
fr = numpy.fft.fftfreq(len(ftrans))

# Steps to filter > 0 values in fr
filter = [] #An empty list

for item in fr: #Loop through the freq array and filter only > 0 values
    if item >= 0:
        filter.append(True)
    else:
        filter.append(False)

freq = fr[filter] """

#%% 
#[:round(ftrans.size/2)]))

""" fourier = numpy.fft.fft(audio)#[:round((audio.size/2))])
n = audio.size 
timestep = 0.1
freq = abs(numpy.fft.fftfreq(n, d=timestep))

audio_trans = abs(fourier)#[:(round(n/20))])
plt.plot(freq, audio_trans)
plt.show()
 """


""" #%% 
plt.plot(freq, 2*(abs(ftrans_pos)))
plt.show() """

#%% 






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


# %% 
# Shady References
#* https://www.youtube.com/watch?v=17cOaqrwXlo
#* https://www.dummies.com/programming/python/performing-a-fast-fourier-transform-fft-on-a-sound-file/
#* https://www.youtube.com/watch?v=17cOaqrwXlo


# %%
