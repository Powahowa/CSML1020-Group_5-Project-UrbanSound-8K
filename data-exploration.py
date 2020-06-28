# %% [markdown] 
# # Feature Exploration
# ## Imports
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import pandas as pd

# Import custom module containing useful functions
import sonicboom

# %% [markdown]
# ## Load the audio files and combine metadata with the paths
file_data = sonicboom.init_metadata('./data/UrbanSound8K/')

CLASSES = list(file_data['class'].unique())
NUM_CLASSES = len(CLASSES)
FOLDS = list(file_data['fold'].unique())
NUM_FOLDS = len(FOLDS)

file_data.rename(columns={0: "path"}, inplace=True)

file_data

# %% [markdown]
# ## Overview of what the data looks like

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
plt.show()

# %% [markdown]
# ### See how Audio Length/Duration is distributed
# file_data = file_data.groupby('class', as_index=False).apply(lambda x: x.sample(10))
# NOTE: above commented sampling down messes with librosa call for get_duration
audio_file = []
sampling_rate = []
duration = []

audio_file, sampling_rate = sonicboom.load_audio(file_data['path'])

file_data['raw_features'] = audio_file
file_data['sampling_rate'] = sampling_rate
file_data['duration'] = [
    librosa.get_duration(file_data['raw_features'][i], 
                         file_data['sampling_rate'][i])
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
