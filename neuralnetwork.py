import numpy as np

class NeuralNetwork:
    def __init__(self, layer_lengths):
        """ Initializes the layers and weights of the neural network.
            layer_lengths: list of lengths for each layer """

        # initialize layers
        self.layer_lengths = layer_lengths
        self.layers = []
        for length in layer_lengths:
            self.layers.append(np.zeros((length, 1)))

        # randomly initialize weights
        self.weights = []
        init_range = 0.12
        for layer in range(len(layer_lengths) - 1):
            weightMatrix = np.random.rand(layer_lengths[layer + 1],\
                           layer_lengths[layer] + 1)
            weightMatrix = weightMatrix * 2 * init_range - init_range
            self.weights.append(weightMatrix);

        # initialize weight gradients
        self.gradients = []
        for layer in range(len(layer_lengths) - 1):
            gradientMatrix = np.zeros((layer_lengths[layer + 1],\
                             layer_lengths[layer] + 1))
            self.gradients.append(gradientMatrix);

    def set_data(self, data):
        """ Sets the data to be used when training or computing the cost
            of the neural network.
            data: numpy.array of data, where each row consists of the
                  feature vector for a single example """
        self.data = data

    def set_labels(self, labels):
        """ Sets the labels to be used when training or computing the cost
            of the neural network.
            labels: numpy.array of labels, where each row consists of the
                    label for its corresponding data example """
        self.labels = labels

    def cost_function(self, reg):
        """ Calculates the weight gradients and returns the regularized cost
            of the neural network based on the current data set and weights.
            reg: regularization factor """
        cost = 0
        for matrix in self.gradients:
            matrix *= 0

        m = len(self.data) # amount of data examples
        X = np.hstack((np.ones((m, 1)), self.data)) # add bias units to data

        # process each data example
        for i in range(m):
            # perform forward propagation / compute activations
            a = [] # list of activations columns
            a.append(X[i, :][None].T) # first activation column is the input
            # compute activation columns of hidden layers
            for layer in range(1, len(self.layer_lengths) - 1):
                column = np.vstack((1, self.sigmoid(self.weights[layer - 1] @\
                         a[layer - 1])))
                a.append(column)
            # compute last activation column / hypothesis
            h = self.sigmoid(self.weights[-1] @ a[-1])

            # accumulate cost for data examples
            y = np.eye(self.layer_lengths[-1])[:, self.labels[i]]
            cost += sum(-y * np.log(h) - (1 - y) * np.log(1- h))

            # perform backpropagation
            d = h - y # delta "error" column
            self.gradients[-1] += d @ a[-1].T
            for x in range(1, len(self.layer_lengths) - 1):
                d = (self.weights[-x].T @ d * a[-x] * (1 - a[-x]))[1:]
                self.gradients[-x - 1] += d @ a[-x - 1].T

        # calculate cost with regularization
        weight_sum = 0
        for matrix in self.weights:
            weight_sum += np.sum(matrix[:, 1:] * matrix[:, 1:])
        cost = cost / m + weight_sum * reg / (2 * m)

        # calculate gradients with regularization
        for matrix in self.gradients:
            matrix = matrix / m
            matrix[:, 1:] += matrix[:, 1:] * reg / m

        for x in range(len(self.gradients)):
            self.gradients[x] = self.gradients[x] / m
            self.gradients[x][:, 1:] += self.weights[x][:, 1:] * reg / m

        return np.sum(cost)

    def train(self, iterations, alpha, reg, test_data, test_labels):
        """ Performs gradient descent.
            iterations: the number of times to perform gradient descent
            alpha: the learning rate for gradient descent
            reg: regularization factor """
        train_plot = []
        test_plot = []
        for i in range(iterations):
            temp_data = self.data
            temp_labels = self.labels
            self.set_data(test_data)
            self.set_labels(test_labels)
            test_plot.append(self.cost_function(0))
            self.set_data(temp_data)
            self.set_labels(temp_labels)
            train_plot.append(self.cost_function(0))
            #print(self.cost_function(reg))
            self.cost_function(reg)
            for matrix in range(len(self.weights)):
                self.weights[matrix] -= alpha * self.gradients[matrix]
        return (train_plot, test_plot)

    @staticmethod
    def sigmoid(x):
        """ Computes the sigmoid activation function
            x: input to the activation function """
        return 1 / (1 + np.exp(-x))

    def accuracy(self):
        """ Prints out the accuracy of the neural network:
        # examples correctly identified / # total examples """
        correct = 0
        m = len(self.data) # amount of data examples
        X = np.hstack((np.ones((m, 1)), self.data)) # add bias units to data

        # process each data example
        for i in range(m):
            # perform forward propagation / compute activations
            a = [] # list of activations columns
            a.append(X[i, :][None].T) # first activation column is the input
            # compute activation columns of hidden layers
            for layer in range(1, len(self.layer_lengths) - 1):
                column = np.vstack((1, self.sigmoid(self.weights[layer - 1] @\
                         a[layer - 1])))
                a.append(column)
            # compute last activation column / hypothesis
            h = self.sigmoid(self.weights[-1] @ a[-1])
            if np.argmax(h) == self.labels[i]:
                correct += 1
            #else:
                #print(np.argmax(h), self.labels[i])

        return correct / m

    def predict(self):
        """ Prints out the accuracy of the neural network:
        # examples correctly identified / # total examples """
        X = np.hstack((np.ones((1, 1)), self.data)) # add bias units to data

        # perform forward propagation / compute activations
        a = [] # list of activations columns
        a.append(X[0, :][None].T) # first activation column is the input
        # compute activation columns of hidden layers
        for layer in range(1, len(self.layer_lengths) - 1):
            column = np.vstack((1, self.sigmoid(self.weights[layer - 1] @\
                     a[layer - 1])))
            a.append(column)
        # compute last activation column / hypothesis
        h = self.sigmoid(self.weights[-1] @ a[-1])
        return np.argmax(h)

    def numericalGrad(self, reg):
        """ Test function to check gradients. """
        numgrad = []
        for x in range(len(self.weights)):
            numgrad.append(self.weights[x].copy())
        e = 0.0001
        for i in range(len(self.weights)):
            for (x,y), value in np.ndenumerate(self.weights[i]):
                 self.weights[i][x][y] -= e
                 loss1 = self.cost_function(reg)
                 self.weights[i][x][y] += 2*e
                 loss2 = self.cost_function(reg)
                 numgrad[i][x][y] = (loss2 - loss1) / (2*e)
                 self.weights[i][x][y] -= e
        return numgrad
