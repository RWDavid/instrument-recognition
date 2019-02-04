import numpy as np
from numpy.fft import fft
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
from neuralnetwork import NeuralNetwork

file_path = input("Enter path to .mp3: ")

# set up neural network with 50 inputs, 30 hidden units, and 5 classes
classes = 3
nn = NeuralNetwork([50, 30, classes])
nn.weights[0] = np.genfromtxt('weights/weights0.csv', delimiter=',')
nn.weights[1] = np.genfromtxt('weights/weights1.csv', delimiter=',')

# extract audio / related information
audio = AudioSegment.from_mp3(file_path)
sample_rate = audio.frame_rate
samples = audio.get_array_of_samples()
l_samples = samples[::2]
r_samples = samples[1::2]
fft_size = 16384

# set up plot
fig, ax = plt.subplots()
data = np.arange(classes)
data[0] = 100;
line, = ax.plot(data)
plt.pause(0.001)

# begin playback
_play_with_simpleaudio(audio)
tstart = time.time()

clarinet = 0
trumpet = 0
violin = 0

# update plot
while True:
    # retrieve current sample
    tnow = time.time() - tstart
    current = int(tnow * sample_rate)

    # break if there are not enough samples (end of song)
    if current + fft_size >= len(l_samples):
        break

    # set top frequency in frequency range
    freq_top = 4000

    # set the number of bins / partitions
    bins = 50

    # determine partitions in frequency range
    x = int(np.ceil(freq_top * fft_size / sample_rate))
    x = x - (x % bins)

    # isolate desired frequency range and perform fft
    data = abs(fft(l_samples[current:current + fft_size])[:x]) +\
           abs(fft(r_samples[current:current + fft_size])[:x])

    # take average over each bin / partition
    a = np.arange(data.size) // (x // bins)
    data = np.bincount(a, data) / np.bincount(a)

    # normalize amplitudes to range [0, 1]
    data = data / max(data)
    nn.set_data(data[None])

    # plot predictions
    #print(nn.predict())
    if np.argmax(nn.predict()) == 0:
        print("clarinet")
        clarinet += 1
    elif np.argmax(nn.predict()) == 1:
        print("trumpet")
        trumpet += 1
    else:
        print("violin")
        violin += 1
    line.set_ydata(nn.predict() * 100)
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Clarinet: " + str(clarinet))
print("Trumpet: " + str(trumpet))
print("Violin: " + str(violin))

def main():
    pass

if __name__ == '__main__':
    main()
