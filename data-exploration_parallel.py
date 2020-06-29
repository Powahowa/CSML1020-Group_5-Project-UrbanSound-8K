# %% [markdown] 
# # Data Exploration
# ## Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
%matplotlib inline
plt.style.use('ggplot')

# Import custom module containing useful functions
import sonicboom

#Parallel libraries
from joblib import Parallel, delayed


# %% [markdown]
# ## Load the audio files and combine metadata with the paths
file_data = sonicboom.init_data('./.data/UrbanSound8K/')

CLASSES = list(file_data['class'].unique())
NUM_CLASSES = len(CLASSES)
FOLDS = list(file_data['fold'].unique())
NUM_FOLDS = len(FOLDS)

file_data.rename(columns={0: "path"}, inplace=True)

file_data

# %% [markdown]
# ## Overview of distributions of the data

# %% [markdown]
# ### Plot the overall counts of the categories
ax = sns.countplot(x='class', data=file_data)
ax.set_title('Overall count of audio files by category')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
)
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ### Plot the counts of the categories by fold
facets = sns.catplot(
    data=file_data,
    col_wrap=5,
    x='class',
    kind='count',
    col='fold'
)
facets.set_xticklabels(rotation=45, horizontalalignment='right')
facets.set_titles('Fold {col_name}')
plt.xlabel('Class')
plt.ylabel('Count')
plt.suptitle('Counts of categories by fold', y=1.05)
plt.show()


# %% [markdown]
# ### See how Audio Length/Duration is distributed

#sample down
#file_data = file_data.groupby('class', as_index=False).apply(lambda x: x.sample(10))
 
def audio_load_parallel(f):
        y, sr = sonicboom.load_audio(f)
        return y, sr

file_data['raw_audio_tuple'] = Parallel(n_jobs=-1)(delayed(audio_load_parallel)(f) for f in file_data['path'])
file_data[['raw_features', 'sample_rate']] = pd.DataFrame(file_data['raw_audio_tuple'].tolist(), index=file_data.index) 
file_data = file_data.drop(columns=['raw_audio_tuple'])

file_data['duration'] = [
    librosa.get_duration(file_data['raw_features'].iloc[i], 
                         file_data['sample_rate'].iloc[i])
    for i in range(file_data.shape[0])
]

# %% [markdown]
# #### In particular, see the distribution of clip length vs class
ax = sns.boxplot(x="class", y="duration", data=file_data)
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
)
ax.set_title('Distribution of Clip Length vs Class')
plt.xlabel('Class')
plt.ylabel('Duration (seconds)')
plt.show()

# %% [markdown]
# #### Also, see the count of various clip lengths
ax = sns.distplot(file_data['duration'], bins=20, kde=False)
ax.set_title('Count of Clip Lengths')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.show()

# %%
# ## Plot Waves and Spectrogram
# ### Plot waves, one for each class
from IPython.display import Audio
plt.figure(figsize=(20,60), dpi=600)
plt.tight_layout()
for i in range(len(CLASSES)):
    selection = file_data[file_data['class'] == CLASSES[i]][:1].reset_index()
    plt.subplot(10,1,i+1)
    librosa.display.waveplot(selection['raw_features'][0], 
                             sr=selection['sample_rate'][0])
    plt.title(selection['class'][0])
plt.suptitle('Wave Plot for each class', y=0.895, fontsize=16)
plt.show()
# Audio(selection['raw_features'][0], rate=selection['sample_rate'][0])

#%%
# ### Plot spectrogram, one for each class
plt.figure(figsize=(20,60))
plt.tight_layout()
for i in range(len(CLASSES)):
    selection = file_data[file_data['class'] == CLASSES[i]][:1].reset_index()
    plt.subplot(10,1,i+1)
    plt.specgram(selection['raw_features'][0], 
             Fs=selection['sample_rate'][0])
    plt.title(selection['class'][0])
plt.suptitle('Spectrogram for each class', y=0.895, fontsize=16)
plt.show()