import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy
import numpy as np
from six.moves import cPickle
import platform
import cv2
sns.set(color_codes=True)


def load_mnist(validation_ratio=0.2):
    X_train = idx2numpy.convert_from_file('Dataset/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('Dataset/train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('Dataset/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('Dataset/t10k-labels.idx1-ubyte')

    train_images = []
    for i in range(X_train.shape[0]):
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)

    test_images = []
    for i in range(X_test.shape[0]):
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)

    indices = np.random.permutation(train_images.shape[0])
    len_validation = int(validation_ratio * len(indices))
    training_idx, validation_idx = indices[len_validation:], indices[:len_validation]
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return cPickle.load(f)
    elif version[0] == '3':
        return cPickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('Images/training_accuracy.jpg')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('Images/learning_curve.jpg')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()


def to_gray(image_name):
    image = cv2.imread(image_name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', image)
    cv2.imwrite(image_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
