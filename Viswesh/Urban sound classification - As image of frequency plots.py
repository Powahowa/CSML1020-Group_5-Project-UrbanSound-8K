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



# %% [markdown]
# Read in the meta-data file
meta = pandas.read_csv('./data/UrbanSound8K/metadata/UrbanSound8K.csv')
meta_fold1 = meta[meta.fold == 1]

# %% [markdown]
""" The idea is to read all sound files found in the meta data list, put them through a fourier
transform and save the images of the transformed signal to perform classification on """

#%% 
# # Set directory to read files from 
src = './data/UrbanSound8K/audio'

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

# Loop through each folder & read all files
""" for i in range(0,len(fold_df)):
    fold = fold_df.iloc[i,0]
    path = '{}/{}'.format(src, ''.join(fold)) """

path = src + '/fold1'

#%%
#Read all files in each path folder iteration
files = glob(path + '/*.wav')
for j in range(0,(len(files))):
    audio, sfreq = librosa.load(files[j])        
    raw_list.append(audio)
    filename.append(os.path.basename(files[j]))
    
    #FFT
    ftrans = numpy.fft.fft(audio) #[:round((audio.size/2))])
    ftrans_pos = ftrans[:round(ftrans.size/2)]
    fr = numpy.fft.fftfreq(len(ftrans))

    # Steps to filter > 0 values in fr
    filter = [] #An empty list for filtering

    for item in fr: #Loop through the freq array and filter only > 0 values
        if item >= 0:
            filter.append(True)
        else:
            filter.append(False)

    fft_freq.append(fr[filter])

# %% 
temp_df = pandas.DataFrame({'slice_file_name': filename, 'fft_freq_array': fft_freq})    

# %%
sound_df = meta_fold1.merge(temp_df, on = 'slice_file_name', how = 'left')

# %%

    



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



# %%
