import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment

def preprocess(directory_path, label, group = False):
    """ Performs the FFT on all mp3 files in the specified directory.
    directory_path: str path to directory (include final forward slash)
    label: the (integer) label to be appended to each data example
    group: option to group data examples into one .dat file """
    file_paths = glob.glob(directory_path + "*")
    count = 0

    os.mkdir(directory_path[:-1] + "_data/")

    if group:
        filename = os.path.abspath(".") + "/" + directory_path[:-1] +\
                   "_data/data.dat"
        dataFile = open(filename, "w")

    for file_path in file_paths:
        audio = AudioSegment.from_mp3(file_path)
        sample_rate = audio.frame_rate
        raw_data = audio.raw_data
        y = np.fromstring(raw_data, dtype=np.int16)
        samples = 16000
        y = y[:samples]
        y = np.abs(np.fft.fft(y)[:len(y)//8]) # frequencies range from 0-5500
        a = np.arange(y.size) // 20 # 100 bins of length 20
        y = np.bincount(a, y) / np.bincount(a) # partition into 100 bins
        y = y / max(y)

        if group:
            dataFile.write(' '.join(str(x) for x in y) +\
                           ' ' + str(label) + '\n')
        else:
            filename = os.path.abspath(".") + "/" + directory_path[:-1] +\
                       "_data/" + str(count) + ".dat"
            dataFile = open(filename, "w")
            dataFile.write(' '.join(str(x) for x in y) +\
                           ' ' + str(label) + '\n')
            dataFile.close()

        count += 1

    if group:
        dataFile.close()
