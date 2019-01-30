import numpy as np
from numpy.fft import fft
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

file_path = input("Enter path to .mp3: ")

# extract audio / related information
audio = AudioSegment.from_mp3(file_path)
sample_rate = audio.frame_rate
samples = audio.get_array_of_samples()
l_samples = samples[::2]
r_samples = samples[1::2]

# set up plot
fig, ax = plt.subplots()
data = np.arange(512)
data[0] = 8000;
line, = ax.plot(data)
plt.pause(0.001)

# begin playback
_play_with_simpleaudio(audio)
tstart = time.time()

# update plot
while True:
    # retrieve current sample
    tnow = time.time() - tstart
    current = int(tnow * sample_rate)

    # break if there are not enough samples (end of song)
    if current + 8192 >= len(l_samples):
        break

    # perform fft
    data = abs(fft(l_samples[current:current + 8192])) / 8192 +\
           abs(fft(r_samples[current:current + 8192])) / 8192
    data = np.array(data[:512])

    # plot amplitudes
    line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
