import numpy as np
import random
import glob
from pydub import AudioSegment
import matplotlib.pyplot as plt

def create_data():
    plot = "Y" == input("Show FFT visualizations of data? (Y/N) ")
    directory_num = int(input("How many directories to process? "))
    data_list = []
    for x in range(directory_num):
        directory_path = input("Enter directory for label " + str(x) + ": ")
        data_list.extend(preprocess(directory_path, x, plot))
    random.shuffle(data_list)

    # calculate 60-20-20 split of data between train/validation/test sets
    total = len(data_list)
    train_num = total * 6 // 10
    validation_num = total * 2 // 10

    # create and write train/validation/test sets
    trainFile = open("train.dat", "w")
    for x in data_list[:train_num]:
        trainFile.write(x)
    trainFile.close()

    validationFile = open("validation.dat", "w")
    for x in data_list[train_num:train_num + validation_num]:
        validationFile.write(x)
    validationFile.close()

    testFile = open("test.dat", "w")
    for x in data_list[train_num + validation_num:]:
        testFile.write(x)
    testFile.close()

def preprocess(directory_path, label, plot):
    """ Performs the FFT on all mp3 files in the specified directory and
    returns a list of data features + labels.
    directory_path: str path to directory (no ending forward slash)
    label: the (integer) label to be appended to each data example
    plot: boolean condition whether to visualize FFT
    (actual + data representation) """
    # obtain file paths for each data example
    file_paths = glob.glob(directory_path + "/*")
    data_list = []

    # set top frequency in frequency range
    freq_top = 4000

    # set the number of bins / partitions
    bins = 50

    # process each audio file
    for file_path in file_paths:
        # extract audio / related information
        audio = AudioSegment.from_mp3(file_path)
        sample_rate = audio.frame_rate
        raw_data = audio.raw_data
        y = np.fromstring(raw_data, dtype=np.int16)
        samples = len(y)

        # determine partitions in frequency range
        freq_res = sample_rate / samples
        x = int(np.ceil(freq_top * samples / sample_rate))
        x = x - (x % bins)

        # isolate desired frequency range
        y = np.abs(np.fft.fft(y)[:x])

        # take average over each bin / partition
        a = np.arange(y.size) // (x // bins)
        y = np.bincount(a, y) / np.bincount(a)

        # normalize amplitudes to range [0, 1]
        y = y / max(y)
        data_list.append(' '.join(str(i) for i in y) + ' ' + str(label) + '\n')

        # plot
        if plot:
            print(file_path)
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            x_var = np.arange(len(y)) * freq_res * (x // bins)
            ax1.plot(x_var, y)

            y = np.fromstring(raw_data, dtype=np.int16)
            y = np.abs(np.fft.fft(y)[:x])
            y = y / max(y)
            x_var = np.arange(len(y)) * freq_res
            ax2.plot(x_var, y)

            plt.show()

    return data_list

def test_audio(file_path):
    # set top frequency in frequency range
    freq_top = 4000

    # set the number of bins / partitions
    bins = 50

    # extract audio / related information
    audio = AudioSegment.from_mp3(file_path)
    sample_rate = audio.frame_rate
    raw_data = audio.raw_data
    y = np.fromstring(raw_data, dtype=np.int16)
    samples = len(y)

    # determine partitions in frequency range
    freq_res = sample_rate / samples
    x = int(np.ceil(freq_top * samples / sample_rate))
    x = x - (x % bins)

    # isolate desired frequency range
    y = np.abs(np.fft.fft(y)[:x])

    # take average over each bin / partition
    a = np.arange(y.size) // (x // bins)
    y = np.bincount(a, y) / np.bincount(a)

    # normalize amplitudes to range [0, 1]
    y = y / max(y)
    return ' '.join(str(i) for i in y)
