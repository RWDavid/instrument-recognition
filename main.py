import numpy as np
from neuralnetwork import NeuralNetwork

def main():
    pass

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

# create neural network with 100 inputs, 30 hidden units, and 5 classes
nn = NeuralNetwork([100, 30, 5])
nn.set_data(train_data)
nn.set_labels(train_labels)

if __name__ == '__main__':
    main()
