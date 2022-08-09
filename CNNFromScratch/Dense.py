from Softmax import *
import numpy as np


class Dense:
    def __init__(self, name, nodes, num_classes):
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases = np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        output = np.dot(input, self.weights) + self.biases
        self.last_input = input
        self.last_output = output
        return softmax(output)

    def backward(self, din, learning_rate=0.005):
        for i, gradient in enumerate(din):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_output)
            dout_dt = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            dout_dt[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)
            dt = gradient * dout_dt
            dout = self.weights @ dt
            self.weights -= learning_rate * (np.transpose(self.last_input[np.newaxis]) @ dt[np.newaxis])
            self.biases -= learning_rate * dt
            return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)