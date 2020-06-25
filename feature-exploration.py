# %% [markdown] 
# # Feature Exploration
# ## Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import custom module containing useful functions
import sonicboom

# %% [markdown]
# ## Load the data files
file_data = sonicboom.init_data('../data/UrbanSound8K/')

CLASSES = list(file_data['class'].unique())
NUM_CLASSES = len(CLASSES)
FOLDS = list(file_data['fold'].unique())
NUM_FOLDS = len(FOLDS)

# %% [markdown]
# ## Overview of what the data looks like

# %% [markdown]
# ### Show the metadata for the audio files
file_data

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

# %%
