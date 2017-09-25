import numpy as np
import h5py
import os


def load_dataset():
    """
    Load all of our data sets that will be used throughout the model (neuron) creationg and training process

    Return:
    train_set_x_orig -- The raw image data used for training our algorithm
    train_set_y_orig -- The raw values for if a training example has a cat in it or not

    test_set_x_orig -- The raw image data used for testing the accuracy of our trained algorithm
    test_set_y_orig -- The raw values for if a test example has a cat in it or not

    classes -- negative log-likelihood cost for logistic regression

    """
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../')

    train_dataset = h5py.File(filename + 'data_sets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(filename + 'data_sets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def unroll_images(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y):
    """
    Take each image matrix and unroll it into a single list of pixels

    Arguments:
    train_set_x_orig -- Matrix of our images for training data
    test_set_x_orig -- Matrix of our images for test data
    train_set_y -- Matrix of our training images correct labels (Cat or no cat)
    test_set_y -- Matrix of our test images correct labels (Cat or no cat)

    Return:
        train_set_x_flatten -- Returns the flattened training set with all the images converted to a list of
    pixels instead of a matrix

    test_set_x_flatten -- Returns the flattened training set with all the images converted to a list of
    pixels instead of a matrix

    """

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    return train_set_x_flatten, test_set_x_flatten


def normalize_dataset(dataset):
    """
    Takes the entire dataset and divides each image by 255. This is a simplified version of normalization.
    Typically you would want to do normalization in the way that makes a vector a unit length vector of 1.
    The division by 255 makes all of the images of various sizes scale to be around the same size.

    Arguments:
    dataset - The dataset of images to normalize.

    Return:
        dataset_scaled - The dataset that is scaled/normalized
    """

    return dataset / 255.


def sigmoid(z):
    """
    Compute the sigmoid of z which constrains a value (z) to be between 0 and 1

    Arguments:
    z -- A numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    # 1 divided by the summation of 1 plus e raised to the power of negative z.
    # np.exp is calling a function which is equal to the constant mathematical value e raised to some power.
    s = 1 / (1 + np.exp(-z))

    return s


def initialize_with_zeros(dimensions=2):
    """
    This function creates a vector of zeros of shape (dimensions, 1) for w and initializes b to 0.

    Argument:
    dimensions -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dimensions, 1))
    b = 0

    assert (w.shape == (dimensions, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b
