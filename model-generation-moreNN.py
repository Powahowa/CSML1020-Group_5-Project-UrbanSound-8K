# %% [markdown]
# # Model Generation

# ## Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
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
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %% [markdown]
# ## Read in the features
filedata = pd.read_pickle('./output/intermediate-data/filedata-mfcc.pkl')

# %% [markdown]
# ## Try traditional ML models
# ### Define and train the models
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

# %% [markdown]
# ### Misclassification Errors
i=0
for model in models:
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

# %% [markdown]
# ## Try Neural Networks
# ### Define feedforward network architecture

# State of the art Swish activation function declaration
from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

%load_ext tensorboard

def get_network():
    input_shape = (40,)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])

    return model

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation
trainaccuracies = []
testaccuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
logdir = './logs/1DMFCC/'
i = 0
for train_index, test_index in kf.split(folds):
    traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
    x_train = np.array(traindata['mfccs'].tolist())
    y_train = np.array(traindata['classID'].tolist())

    testdata = filedata[filedata['fold'] == folds[test_index][0]]
    x_test = np.array(testdata["mfccs"].tolist())
    y_test = np.array(testdata["classID"].tolist())

    # Possibly do mean normalization here on x_train and
    # x_test but using only x_train's mean and std.
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train=(x_train-mean)/std
    x_test=(x_test-mean)/std

    checkpoint = keras.callbacks.ModelCheckpoint('best_%d.h5'%i, monitor='val_loss',
        verbose=1, save_best_only=True)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    tb = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    callbacks_list = [checkpoint, early, tb]

    model = get_network()
    history = model.fit(x_train, y_train, epochs=10,
              use_multiprocessing=True, verbose=0,
              callbacks=callbacks_list, validation_data=(x_test, y_test))
    ltrain, atrain = model.evaluate(x_train, y_train, verbose=0)
    ltest, atest = model.evaluate(x_test, y_test, verbose=0)
    trainaccuracies.append(atrain)
    testaccuracies.append(atest)
    print("Train Loss: {0} | Accuracy: {1}".format(ltrain, atrain))
    print("Test Loss: {0} | Accuracy: {1}".format(ltest, atest))
    i += 1

print("Average Train 10 Folds Accuracy: {0}".format(np.mean(trainaccuracies)))
print("Average Test 10 Folds Accuracy: {0}".format(np.mean(testaccuracies)))

# %%
%tensorboard --logdir logs/fit

# %% [markdown]
# ### Define convolutional network architecture
def get_cnn():
    num_filters = [24,32,64,128] 
    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    input_shape = (41, 60, 2)
    num_classes = 10
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size,
                padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=["accuracy"])
    return model

import librosa
import sonicboom

def extract_features():
    bands=60
    frames=41
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2)
            
    window_size = 512 * (frames - 1)
    features, labels = [], []
    for fn in filedata['path']:
        segment_log_specgrams = []
        sound_clip,sr = librosa.load(fn)
        for (start,end) in _windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal,n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                segment_log_specgrams.append(logspec)
            
        segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
            len(segment_log_specgrams),frames,bands,1)
        segment_features = np.concatenate((segment_log_specgrams, np.zeros(
            np.shape(segment_log_specgrams))), axis=3)
        for i in range(len(segment_features)): 
            segment_features[i, :, :, 1] = librosa.feature.delta(
                segment_features[i, :, :, 0])
        segment_features = tf.convert_to_tensor(segment_features)
        # if len(segment_features) > 0: # check for empty segments 
        features.append(segment_features)
    return features

# %%
filedata = filedata.groupby(
    'class', 
    as_index=False, 
    group_keys=False
).apply(lambda x: x.sample(10))

# %%
fff = extract_features()

# %%
# for i in list(range(len(filedata))):
    # filedata['extrafeatures2'][i] = fff[i].astype(np.float32)
filedata['extrafeatures'] = fff

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation
from sklearn.metrics import accuracy_score
accuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds):
    traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
    x_train = (traindata['extrafeatures2'])
    y_train = tf.convert_to_tensor(traindata['classID'].tolist())

    testdata = filedata[filedata['fold'] == folds[test_index][0]]
    x_test = (testdata['extrafeatures2'])
    y_test = tf.convert_to_tensor(testdata['classID'].tolist())

    model = get_cnn()
    model.fit(x_train, y_train, epochs = 10, batch_size = 24, verbose = 0)
    
    # evaluate on test set/fold
    y_true, y_pred = [], []
    for x, y in zip(x_test, y_test):
        # average predictions over segments of a sound clip
        avg_p = np.argmax(np.mean(model.predict(x), axis = 0))
        y_pred.append(avg_p) 
        # pick single label via np.unique for a sound clip
        y_true.append(np.unique(y)[0]) 
    accuracies.append(accuracy_score(y_true, y_pred))    
print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))

# %%
