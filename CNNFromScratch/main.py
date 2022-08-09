from NetworkModel import *
from CNN import *
from Dense import *
from Utils import *

if __name__ == '__main__':
    num_epochs = 1
    learning_rate = 0.01
    validate = 1
    regularization = 0
    verbose = 1
    plot_weights = 1
    plot_correct = 0
    plot_missclassified = 0
    plot_feature_maps = 0

    print('\n--- Loading dataset ---')
    dataset = load_mnist()

    print('\n--- Processing the dataset ---')
    dataset = preprocess(dataset)

    print('\n--- Building the model ---')
    model = Network()
    model.add_layer(Cnn2D(name='conv1', num_filters=8, stride=2, size=3, activation='LeakyReLU'))
    model.add_layer(Cnn2D(name='conv2', num_filters=8, stride=2, size=3, activation='LeakyReLU'))
    model.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))

    print('\n--- Training the model ---')
    model.train(
        dataset,
        num_epochs,
        learning_rate,
        validate,
        regularization,
        plot_weights,
        verbose
    )

    print('\n--- Testing the model ---')
    model.evaluate(
        dataset['test_images'],
        dataset['test_labels'],
        regularization,
        plot_correct,
        plot_missclassified,
        plot_feature_maps,
        verbose
    )