import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork import NeuralNetwork
from preprocess import test_audio

def train_and_plot(nn, iterations, alpha, reg, test_data, test_labels):

    # set up plot
    fig, ax = plt.subplots()
    ind = np.arange(1, 4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('Model Performance: Lambda = 3')

    (train_plot, test_plot) = nn.train_with_plot(iterations, alpha, reg, test_data, test_labels)
    min_epoch = np.argmin(test_plot) + 1
    print("Test cost minimized at epoch " + str(min_epoch))
    epochs = np.arange(len(train_plot))
    plt.plot(epochs, train_plot, label = "Training")
    plt.plot(epochs, test_plot, label = "Cross Validation")
    plt.legend()
    plt.show()

# load train data
train_data = []
train_labels = []
train_file = open("data/train.dat", "r")
for line in train_file:
    train_data.append([float(x) for x in line[:-2].split()])
    train_labels.append(int(line.split()[-1]))
train_file.close()

# load validation data
validation_data = []
validation_labels = []
validation_file = open("data/validation.dat", "r")
for line in validation_file:
    validation_data.append([float(x) for x in line[:-2].split()])
    validation_labels.append(int(line.split()[-1]))
validation_file.close()

# load test data
test_data = []
test_labels = []
test_file = open("data/test.dat", "r")
for line in test_file:
    test_data.append([float(x) for x in line[:-2].split()])
    test_labels.append(int(line.split()[-1]))
test_file.close()

# create np.arrays for data/labels
train_data = np.array(train_data)
train_labels = np.array(train_labels).reshape(len(train_labels), 1)

validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels).reshape(len(validation_labels), 1)

test_data = np.array(test_data)
test_labels = np.array(test_labels).reshape(len(test_labels), 1)

# create neural network with 50 inputs, 30 hidden units, and 3 classes
nn = NeuralNetwork([50, 30, 3])
nn.set_data(train_data)
nn.set_labels(train_labels)
