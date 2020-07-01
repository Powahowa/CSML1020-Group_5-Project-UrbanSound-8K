# %% [markdown]
# Import Dependencies
import numpy
import pandas
import scipy 
from scipy import io
from scipy.io.wavfile import read as wavread
from scipy.fftpack import fft
import librosa
import matplotlib.pyplot as plt 
from glob import glob


# %% 
# Set directory to read files from 
src = './Data/UrbanSound8K/audio'#/fold1'
sound_file = glob(src + '/*/*.wav')

# %% [markdown]
# Read in Data using scipy (sample)

""" testfile = scipy.io.wavfile.read("./Data/UrbanSound8K/audio/fold1/7383-3-0-1.wav")
aud_array = testfile[1]
if aud_array.ndim == 2: 
    aud_array = numpy.mean(testfile[1], axis = 1) #averages the two channels of a stereo signal 
else:
    aud_array = testfile[1]
 """

#%% 
# Read in data using Librosa
audio, sfreq = librosa.load(sound_file[1,4])

# %%
timeline = numpy.arange(0, len(audio)) / sfreq

# %% 
# Manually plot on "time domain"
""" time = numpy.linspace(0, len(aud_array)/testfile[0], num=len(aud_array))
plt.plot(aud_array)
plt.show()
plt.plot(time, aud_array)
plt.show() """

# %% 
# Plot audio over time using Librosa inputs
fig, ax = plt.subplots()
ax.plot(timeline, audio)
ax.set(xlabel = "Time in secs", ylabel = "Amplitude")
plt.show()

# %% 
# Fourier Transform
ftrans = numpy.fft.fft(audio) #[:round((audio.size/2))])
ftrans_pos = ftrans[:round(ftrans.size/2)]
fr = numpy.fft.fftfreq(len(ftrans))

#%%
# Steps to filter > 0 values in fr
filter = [] #An empty list

for item in fr: #Loop through the freq array and filter only > 0 values
    if item >= 0:
        filter.append(True)
    else:
        filter.append(False)

freq = fr[filter]

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


#%% 
plt.plot(freq, 2*(abs(ftrans_pos)))
plt.show()

#%% 






# %% [markdown]
# Code References
#* https://www.youtube.com/watch?v=vJ_WL9aYfNI
#* https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/#:~:text=Glob%20is%20a%20general%20term,pathnames%20matching%20a%20specified%20pattern.


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
