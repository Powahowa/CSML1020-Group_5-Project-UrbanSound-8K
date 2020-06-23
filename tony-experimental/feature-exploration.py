# %% [markdown] 
# # Feature Exploration
# ## Imports
import sonicboom
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %% [markdown]
# ## Define some decorator functions
import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        startTime = time.perf_counter()
        value = func(*args, **kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f'Finished {func.__name__!r} in {runTime:.4f} secs')
        return value
    return wrapper

# %% [markdown]
# ## Load the data files
file_data = sonicboom.init_read_audio('../data/UrbanSound8K/')

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
ax = sns.countplot(x="class", data=file_data)
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'
)

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

# %%
