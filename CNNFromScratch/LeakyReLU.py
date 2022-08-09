def leakyReLU(x, alpha=0.001):
    return x * alpha if x < 0 else x


def leakyReLU_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1