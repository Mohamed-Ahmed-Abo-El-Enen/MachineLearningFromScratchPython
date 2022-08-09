from LeakyReLU import *
import numpy as np


class FullyConnected:
    def __init__(self, name, nodes1, nodes2, activation):
        self.name = name
        self.weights = np.random.randn(nodes1, nodes2) * 0.1
        self.biases = np.zeros(nodes2)
        self.activation = activation
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

        if self.activation == 'LeakyReLU':
            self.activation_function = np.vectorize(leakyReLU)
            self.activation_function_gradient = np.vectorize(leakyReLU_derivative)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        output = np.dot(input, self.weights) + self.biases

        if self.activation == 'LeakyReLU':
            self.activation_function(output)

        self.last_input = input
        self.last_output = output

        return output

    def backward(self, din, learning_rate=0.005):
        if self.activation == 'LeakyReLU':
           self.activation_function_gradient(din)

        self.last_input = np.expand_dims(self.last_input, axis=1)
        din = np.expand_dims(din, axis=1)

        dw = np.dot(self.last_input, np.transpose(din))
        db = np.sum(din, axis=1).reshape(self.biases.shape)

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        dout = np.dot(self.weights, din)
        return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)

