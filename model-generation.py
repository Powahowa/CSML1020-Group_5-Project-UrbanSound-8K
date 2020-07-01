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
import matplotlib as plt
plt.style.use('ggplot')

# %% [markdown]
# ## Read in the features
filedata = pd.read_csv('./output/intermediate-data/filedata-mfcc.csv')

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

# %% [markdown]
# ## Try Neural Network
# ### Define feedforward network architecture
def get_network():
    input_shape = (40,)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"])

    return model

# %% [markdown]
# ### Train and evaluate via 10-Folds cross-validation
accuracies = []
folds = np.array(list(range(1,11)))
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds):
    traindata = filedata[filedata['fold'].isin(list(folds[train_index]))]
    x_train = pd.DataFrame(traindata['mfccs'], 
        columns=list(range(1, 40+1)))
    y_train = pd.Series(traindata['classID'].tolist())

    testdata = filedata[filedata['fold'] == folds[test_index][0]]
    x_test = pd.DataFrame(testdata["mfccs"],
        columns=list(range(1, 40+1)))
    y_test = pd.Series(testdata["classID"].tolist())

    # Possibly do mean normalization here on x_train and
    # x_test but using only x_train's mean and std.

    model = get_network()
    model.fit(x_train, y_train, epochs = 5, verbose = 1)
    l, a = model.evaluate(x_test, y_test, verbose = 1)
    accuracies.append(a)
    print("Loss: {0} | Accuracy: {1}".format(l, a))

print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))