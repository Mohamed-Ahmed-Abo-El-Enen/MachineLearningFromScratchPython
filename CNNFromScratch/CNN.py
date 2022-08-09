from LeakyReLU import *
import numpy as np


class Cnn2D:
    def __init__(self, name, num_filters=16, size=3, stride=1, activation=None):
        self.name = name
        self.filters = np.random.randn(num_filters, size, size) * 0.1
        self.stride = stride
        self.size = size
        self.activation = activation
        self.last_input = None

        if self.activation == 'LeakyReLU':
            self.activation_function = np.vectorize(leakyReLU)
            self.activation_function_gradient = np.vectorize(leakyReLU_derivative)

    def forward(self, image):
        self.last_input = image
        input_dimension = image.shape[1]
        output_dimension = int((input_dimension - self.size) / self.stride) + 1
        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))

        for f in range(self.filters.shape[0]):
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = image[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        if self.activation == 'LeakyReLU':
            self.activation_function(out)
        return out

    def backward(self, din, learn_rate=0.005):
        input_dimension = self.last_input.shape[1]

        if self.activation == 'LeakyReLU':
           self.activation_function_gradient(din)

        dout = np.zeros(self.last_input.shape)
        dfilt = np.zeros(self.filters.shape)

        for f in range(self.filters.shape[0]):
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.filters -= learn_rate * dfilt

        return dout

    def get_weights(self):
        return np.reshape(self.filters, -1)