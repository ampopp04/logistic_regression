import os
import numpy as np
import scipy

from scipy import ndimage
import matplotlib.pyplot as plt

from logistic_regression.utils.data_utils import sigmoid
from logistic_regression.utils.data_utils import initialize_with_zeros


def propagate(w, b, X, Y):
    """
    A single iteration of backpropagation

    Arguments:
    w -- weights, a numpy matrix of size (num_px * num_px * 3, 1)
    b -- our hidden b constant bias value needed for our model
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w
    db -- gradient of the loss with respect to b

    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # This is actually running a real prediction on our model
    # A contains all the answers the model made across all the images in X
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation

    # Compute the cost, this isn't really important it's more used for graphing the learning over time
    cost = -1. / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    # Compute the derivative of loss with respect to our sigmoid function
    dz = A - Y

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.divide(np.dot(X, dz.T), m)
    db = np.divide(np.sum(dz), m)

    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias constant b needed for our model
    X -- images vector of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule or what we refer to as alpha
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    #  Try to find the best optimized model (neuron) within the num_iterations ('time-limit')
    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Here we have the full Gradient Descent Algorithm in action
        # The w and b values will slowly be optimized to minimize loss over each iteration of i
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias constant value
    X -- image vector of shape (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] <= .5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    trained_model -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    trained_model = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return trained_model


def run_trained_model_on_image(image_name, train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes,
                               num_iterations=2000, learning_rate=0.005, print_cost=True):
    """
    Creates and trains a neuron using logistic regression and gradient descent. After training, it will then
    make a prediction on the image that you provide as to whether there is a cat in it or not.

    Arguments:
    image_name -- The name of the image held within the data_sets directory Ex. "MyCatOrNotPic.jpg

    train_set_x -- Images used to train our model, they are unrolled(List of pixels), and normalized.
    train_set_y -- The labels that define if there is a cat or not in a specific image contained in train_set_x

    test_set_x -- Images used to test our model after training to determine it's accuracy on unseen images
    test_set_y -- The labels that define if there is a cat or not in a specific image contained in test_set_x

    num_px -- The width/height of our picture assuming it is a square.
    classes -- The classes of objects in our data set, which in this case is either cat or not-cat images

    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimization
    print_cost -- Set to true to print the current cost every 100 iterations of optimization


    Returns:
     Nothing - Will print the result of if it thought there was a cat or not and then it will show the picture
    """

    # Create and train our model (neuron)
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost)

    #  Get a reference to our data_sets directory
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../')

    # We load the image
    image = np.array(ndimage.imread(filename + "data_sets/images/" + image_name, flatten=False))

    #  Reshape the image such that it is not a 3-Dimensional array but a single row of pixels
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T

    # NOTE MIGHT WANT TO TRY NORMALIZING THE IMAGE HERE

    # Ask the neuron, do you think there is a cat in the picture?
    prediction_result = predict(d["w"], d["b"], my_image)

    #  Print out what the model predicted
    print("y = " + str(np.squeeze(prediction_result)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(prediction_result)),].decode("utf-8") + "\" picture.")

    #  Show the image that the neuron looked at to make it's prediction
    plt.imshow(image)
    plt.show(block=True)
