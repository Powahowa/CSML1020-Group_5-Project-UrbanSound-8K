# %% [markdown]
# # Model Generation

# ## Imports
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
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
filedata = pd.read_pickle('./output/intermediate-data/filedata-librosaFeatures.pkl')

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

#this code goes through and separates out each  elements of each feature array into separate columns
# DON'T CHANGE IT
mfccs = pd.DataFrame(filedata['mfccs'].iloc[x] for x in range(len(filedata)))
melSpec = pd.DataFrame(filedata['melSpec'].iloc[x] for x in range(len(filedata)))
stft = pd.DataFrame(filedata['stft'].iloc[x] for x in range(len(filedata)))
chroma_stft = pd.DataFrame(filedata['chroma_stft'].iloc[x] for x in range(len(filedata)))
spectral_contrast_stft = pd.DataFrame(filedata['spectral_contrast_stft'].iloc[x] for x in range(len(filedata)))
tonnetz = pd.DataFrame(filedata['tonnetz'].iloc[x] for x in range(len(filedata)))
#visFFT = pd.DataFrame(filedata['visFFT'].iloc[x] for x in range(len(filedata)))

X = pd.concat([mfccs, melSpec, stft, chroma_stft, spectral_contrast_stft, \
    tonnetz], axis=1) 



#X = filedata['melSpec']
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
    # WTF
    plot_learning_curves(X, y, X, y, model)
    # WTF
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
    cm = confusion_matrix(np.argmax(y, axis=1), np.argmax(y_test_pred[i], axis=1))
    cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.title('Confusion Matrix for ' + model_namelist[i], fontsize=14)
    sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
    plt.show()
    i += 1

# %%

print(mfccs.shape)
print(melSpec.shape)
print(stft.shape)
print(chroma_stft.shape)
print(spectral_contrast_stft.shape)
print(tonnetz.shape)
print(visFFT.shape)
print(X.shape)

# %%
