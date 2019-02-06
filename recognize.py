import numpy as np
import sys
from numpy.fft import fft
from pydub import AudioSegment
from neuralnetwork import NeuralNetwork

def main():
    if len(sys.argv) != 2:
        print("Incorrect amount of inputs. Please enter filepath to audio.")
        sys.exit(-1)

    file_path = sys.argv[1]

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
    fft_size = 8192

    # instrument counters
    clarinet = 0
    trumpet = 0
    violin = 0

    # current sample
    current = 0

    # set top frequency in frequency range
    freq_top = 4000

    # set the number of bins / partitions
    bins = 50

    # determine partitions in frequency range
    x = int(np.ceil(freq_top * fft_size / sample_rate))
    x = x - (x % bins)

    while current + fft_size < len(l_samples):
        # isolate desired frequency range and perform fft
        data = abs(fft(l_samples[current:current + fft_size])[:x]) +\
               abs(fft(r_samples[current:current + fft_size])[:x])

        # take average over each bin / partition
        a = np.arange(data.size) // (x // bins)
        data = np.bincount(a, data) / np.bincount(a)

        # normalize amplitudes to range [0, 1]
        data = data / max(data)
        nn.set_data(data[None])

        # record predictions
        if np.argmax(nn.predict()) == 0:
            clarinet += 1
        elif np.argmax(nn.predict()) == 1:
            trumpet += 1
        else:
            violin += 1

        current += sample_rate // 5

    total = clarinet + trumpet + violin

    if (clarinet >= trumpet and clarinet >= violin):
        print("\nPrediction: Clarinet")
        print("Confidence: " + str(clarinet / total))
    elif (trumpet >= clarinet and trumpet >= violin):
        print("\nPrediction: Trumpet")
        print("Confidence: " + str(trumpet / total))
    else:
        print("\nPrediction: Violin")
        print("Confidence: " + str(violin / total))
    print("")

if __name__ == '__main__':
    main()
