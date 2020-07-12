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
# Hyperparameter tuning functions using keras-tuner
import kerastuner as kt
from tensorboard.plugins.hparams import api as hp
from sklearn.metrics import accuracy_score
import librosa
import sonicboom

# %% [markdown]
# ## Read in the features
filedata = pd.read_pickle('./output/intermediate-data/filedata-mfcc-100perclass.pkl')

# %% [markdown]
# ## Try Neural Networks
# ### Define feedforward network architecture

# Function used to define the feedforward neural network
def get_network():
    input_shape = (40,)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation=keras.activations.swish, 
                                input_shape=input_shape))                        
    model.add(keras.layers.Dense(128, activation=keras.activations.swish))
    model.add(keras.layers.Dense(64, activation=keras.activations.swish))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))

    lr_schedule = 1e-4
    # lr_schedule = keras.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=1e-1,
    #     decay_steps=1,
    #     end_learning_rate=0.0001,
    #     power=0.5)
    opt = keras.optimizers.Adam(learning_rate = lr_schedule)

    model.compile(optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])

    return model

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation

# Declare vital variables
trainaccuracies = []
testaccuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
logdir = './logs/1DMFCC/'
i = 0

# Loop through the folds fitting the data
for train_index, test_index in kf.split(folds):
    # Extract the data for train
    traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
    x_train = np.array(traindata['mfccs'].tolist())
    y_train = np.array(traindata['classID'].tolist())

    # Extract the data for test
    testdata = filedata[filedata['fold'] == folds[test_index][0]]
    x_test = np.array(testdata["mfccs"].tolist())
    y_test = np.array(testdata["classID"].tolist())

    # Do mean normalization here on x_train and
    # x_test but using only x_train's mean and std.
    # mean = np.mean(x_train, axis=0)
    # std = np.std(x_train, axis=0)
    # x_train=(x_train-mean)/std
    # x_test=(x_test-mean)/std

    # Checkpoint to continue models, early stopping and tensorboard
    checkpoint = keras.callbacks.ModelCheckpoint(
        logdir + 'best_%d.h5'%i, 
        monitor='val_loss',
        verbose=0, 
        save_weights_only=True, 
        save_best_only=True
    )
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=25
    )
    tb = keras.callbacks.TensorBoard(log_dir=logdir)
    # callbacks_list = [checkpoint, early, tb]
    callbacks_list = [checkpoint, early]

    model = get_network()
    history = model.fit(
        x_train, 
        y_train, 
        epochs=1000,
        use_multiprocessing=True, 
        verbose=0,
        callbacks=callbacks_list, 
        validation_data=(x_test, y_test)
    )

    trainloss, trainacc = model.evaluate(x_train, y_train, verbose=0)
    testloss, testacc = model.evaluate(x_test, y_test, verbose=0)
    trainaccuracies.append(trainacc)
    testaccuracies.append(testacc)
    print(f"Fold: {i}")
    print("Train Loss: {0} | Accuracy: {1}".format(trainloss, trainacc))
    print("Test Loss: {0} | Accuracy: {1}".format(testloss, testacc))
    i += 1

# Out of loop, print average of the results
print("===============================================")
print("Average Train 10 Folds Accuracy: {0}".format(np.mean(trainaccuracies)))
print("Average Test 10 Folds Accuracy: {0}".format(np.mean(testaccuracies)))

# %%
def get_network_hp(hp):
    input_shape = (40,)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="swish", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=1000,
    #     decay_rate=0.9)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
    opt = keras.optimizers.Adam(learning_rate = hp_learning_rate)
    model.compile(optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])

    return model

tuner = kt.Hyperband(get_network_hp,
                     objective = 'val_accuracy', 
                     max_epochs = 10,
                     factor = 3,
                     directory = './logs/hparam_tuning',
                     project_name = 'UrbanSound8K')    

# %%
# HYPERPARAMETER TUNING VERSION OF ABOVE using keras-tuner Hyperband

# Declare vital variables
trainaccuracies = []
testaccuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
logdir = './logs/1DMFCC/'
i = 0

# Loop through the folds fitting the data
for train_index, test_index in kf.split(folds):
    if i==0:
        # Extract the data for train
        traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
        x_train = np.array(traindata['mfccs'].tolist())
        y_train = np.array(traindata['classID'].tolist())

        # Extract the data for test
        testdata = filedata[filedata['fold'] == folds[test_index][0]]
        x_test = np.array(testdata["mfccs"].tolist())
        y_test = np.array(testdata["classID"].tolist())

        # Checkpoint to continue models, early stopping and tensorboard
        checkpoint = keras.callbacks.ModelCheckpoint(
            logdir + 'best_%d.h5'%i, 
            monitor='val_loss',
            verbose=0, 
            save_weights_only=True, 
            save_best_only=True
        )
        early = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=20
        )
        tb = keras.callbacks.TensorBoard(log_dir=logdir)
        # callbacks_list = [checkpoint, early, tb]
        callbacks_list = [checkpoint, early]

        tuner.search(x_train, 
            y_train, 
            epochs = 10, 
            validation_data = (x_test, y_test)
        )

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)
    i += 1

# %%
# Try Hyperparameter tuning with Tensorboard HParams dashboard
%load_ext tensorboard

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([5, 40]))
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 4))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5]))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 3))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adagrad", "adam"]))

HPARAMS = [
    HP_NUM_UNITS,
    HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_DENSE_LAYERS,
    HP_DROPOUT,
    HP_OPTIMIZER,
]

METRICS = [
    hp.Metric(
        "epoch_accuracy", group="validation", display_name="accuracy (val.)",
    ),
    hp.Metric("epoch_loss", group="validation", display_name="loss (val.)",),
    hp.Metric(
        "batch_accuracy", group="train", display_name="accuracy (train)",
    ),
    hp.Metric("batch_loss", group="train", display_name="loss (train)",),
]

METRIC_ACCURACY = [hp.Metric('accuracy', display_name='Accuracy')]

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=HPARAMS,
        metrics=METRIC_ACCURACY,
    )

def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=10, # Run with 1 epoch to speed things up for demo purposes
                callbacks=[
                tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                hp.KerasCallback(logdir, hparams),  # log hparams
            ]
    ) 
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1

print("Done")

# %%
%tensorboard --logdir logs/fit

# %% [markdown]
# ### Define convolutional network architecture

def extract_features(filedata):
    features, labels = [], []
    for fn in filedata['path']:
        sound_clip,sr = librosa.load(fn)
        melspec = librosa.feature.mfcc(sound_clip, n_mfcc=40)
        deltas = librosa.feature.delta(melspec)
        combi = np.dstack((melspec, deltas))
        features.append(combi)
    return features

# %%
# filedata = filedata.groupby(
#     'class', 
#     as_index=False, 
#     group_keys=False
# ).apply(lambda x: x.sample(10))

# %%
fff = extract_features(filedata)
filedata['extrafeatures'] = fff

# %%
testdata = pd.read_pickle('./output/intermediate-data/filedata-librosaFeatures-test.pkl')
ttt = extract_features(testdata)
testdata['extrafeatures'] = ttt
x_test = np.array(testdata['extrafeatures'].tolist())
y_test = np.array(testdata['classID'].tolist())

# %%

def get_cnn():
    num_filters = [24,32,64,128] 
    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    input_shape = (40, 173, 2)
    num_classes = 10
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size, input_shape=input_shape,
                padding="same"))
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
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3), 
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=["accuracy"])
    return model

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation
accuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
trainaccuracies = []
valaccuracies = []
testaccuracies = []
i = 0
logdir = './logs/2DMFCCwDelta/'
num_epochs = 1000
num_waits = 50
verbosity = 0

for train_index, test_index in kf.split(folds):
    traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
    x_train = np.array(traindata['extrafeatures'].tolist())
    y_train = np.array(traindata['classID'].tolist())

    testdata = filedata[filedata['fold'] == folds[test_index][0]]
    x_val = np.array(traindata['extrafeatures'].tolist())
    y_val = np.array(traindata['classID'].tolist())

    # Checkpoint to continue models, early stopping and tensorboard
    checkpoint = keras.callbacks.ModelCheckpoint(
        logdir + 'best_%d.h5'%i, 
        monitor='val_loss',
        verbose=verbosity, 
        save_weights_only=True, 
        save_best_only=True
    )
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=num_waits
    )
    tb = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks_list = [checkpoint, early, tb]
    # callbacks_list = [checkpoint, early]

    model = get_cnn()
    # model.fit(x_train, y_train, epochs = 10, batch_size = 24, verbose = 0)
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        use_multiprocessing=True, 
        verbose=0,
        callbacks=callbacks_list,
        validation_data=(x_val, y_val)
    )
    trainloss, trainacc = model.evaluate(x_train, y_train, verbose=0)
    valloss, valacc = model.evaluate(x_val, y_val, verbose=0)
    testloss, testacc = model.evaluate(x_test, y_test, verbose=0)
    trainaccuracies.append(trainacc)
    valaccuracies.append(valacc)
    testaccuracies.append(testacc)
    print(f"Fold: {i}")
    print("Train Loss: {0} | Accuracy: {1}".format(trainloss, trainacc))
    print("Val Loss: {0} | Accuracy: {1}".format(valloss, valacc))
    print("Test Loss: {0} | Accuracy: {1}".format(testloss, testacc))
    i += 1

# Out of loop, print average of the results
print("===============================================")
print("FINISHED!")
print(f"Number of Epochs per fold: {num_epochs}")
print("Average Train 10 Folds Accuracy: {0}".format(np.mean(trainaccuracies)))
print("Average Val 10 Folds Accuracy: {0}".format(np.mean(valaccuracies)))
print("Average Test 10 Folds Accuracy: {0}".format(np.mean(testaccuracies)))

# %%
