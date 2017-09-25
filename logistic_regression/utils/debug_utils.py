import numpy as np
import matplotlib.pyplot as plt
from logistic_regression.utils.model_utils import model
from logistic_regression.utils.data_utils import normalize_dataset
from logistic_regression.utils.data_utils import unroll_images


def show_image(index, train_set_x_orig, train_set_y, classes):
    """
    Show an image from the given data set and if it contains a cat or not

    Arguments:
    index -- Which image to show
    train_set_x_orig -- The data set of images to pull from using the given index
    train_set_y -- Data set of whether or not the image at the provided image is a cat or not
    classes -- The classes of data held within the data set, this would be either cat or not cat images

    Return:
    Nothing - Shows the image at the specific index providing information about if it is a cat or not

    """

    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")

    plt.show(block=True)


def print_dataset_details(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y):
    """
    Prints to console various stats on the data sets that we will be using to train our neuron

    Arguments:
    train_set_x_orig -- Raw data set of our training images
    test_set_x_orig -- Raw data set of our test images

    train_set_y -- Labels for if a specific image in our training set is a cat or not
    test_set_y -- Labels for if a specific image in our test set is a cat or not

    Return:
    Nothing - Prints stats on the various data sets

    """

    #  Determine how many training and test images we have
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    #  Determine Height/Width of our images
    num_px = train_set_x_orig.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))


def show_incorrect_classification(trained_model, num_px, test_set_x, test_set_y, classes, index=6):
    """
    Take the results of a trained model and shows an image that it incorrectly classified during testing
    This is interesting because it gives you a view of an image that it got wrong

    Arguments:
    trained_model -- A dictionary containing all of the information composing our trained model
    num_px -- Width/height of the pictures used during training/testing

    test_set_x -- Images from our test data set
    test_set_y -- Labels for images in our test data set that denotes if there is a cat in them or not

    classes -- The generic classes of images that we have, specifically in our case cat and not-cat
    index -- Out of all incorrect classifications this index denotes the one to show

    Return:
    Nothing - Shows the image that the model (neuron) got wrong denoted by the provided index
        It also prints out information to the console on what it 'thought'

    """

    # Example of a picture that was wrongly classified.
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
        trained_model["Y_prediction_test"][0, index]].decode("utf-8") + "\" picture.")


def plot_learning_curve(trained_model):
    """
    For each iteration this method will show a graph of how the cost of the model changes

    Arguments:
    trained_model -- A dictionary containing all of the information composing our trained model

    Return:
    Nothing - Shows image of model cost during optimization vs iterations of optimizations

    """

    # Plot learning curve (with costs)
    costs = np.squeeze(trained_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(trained_model["learning_rate"]))
    plt.show()
    plt.show(block=True)


def plot_effect_of_learning_rate(train_set_x, train_set_y, test_set_x, test_set_y, learning_rates = [0.01, 0.001, 0.0001]):
    """
    Plots the cost function over time for various choices you could have picked for learning rates
    This is good for getting an idea of how different learning rates you pick
    might change the behavior of optimization and model accuracy

    Arguments:
    train_set_x -- Images used to train our model, they are unrolled(List of pixels), and normalized.
    train_set_y -- The labels that define if there is a cat or not in a specific image contained in train_set_x

    test_set_x -- Images used to test our model after training to determine it's accuracy on unseen images
    test_set_y -- The labels that define if there is a cat or not in a specific image contained in test_set_x

    learning_rates -- An array of various learning rates to try and then plot the cost vs iteration graphs for each

    Return:
    Nothing - Plots model cost vs iterations for different choices of a learning rate. This gives you insight
    into how the learning rate (alpha) can change the optimization process or the outcome of the model

    """

    #  Container for each model that is produced, one for each learning rate
    models = {}

    #  Generate and train the models for the provided learning rate
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    # Graph the learning rate results of cost vs iterations onto a single window
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    # Configure and show the plot
    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    plt.show(block=True)


def show_all_details_of_learning_and_optimization_process(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y, classes):
    """
       Break down the data loading, model creation, and training processes in a way such that we can show interesting statistics
       This is useful for seeing various aspects of the process printed in greater detail to the console or graphed in windows.
       If a graph is shown that will halt the processing of code until you close that specific graph window, then it will proceed.

       Arguments:
       train_set_x_orig -- Matrix of our images for training data
       test_set_x_orig -- Matrix of our images for test data
       train_set_y -- Matrix of our training images correct labels (Cat or no cat)
       test_set_y -- Matrix of our test images correct labels (Cat or no cat)
       classes -- Denotes the types of data we have in our images, for our example they are classes of either cat or non-cat images

       Return:
           Nothing - Shows stats in the console and produces various graphs for informational purposes

       """

    # Show a picture from the data set
    # Code execution will pause until you close the image
    index = 10  # Pick a random image to see
    show_image(index, train_set_x_orig, train_set_y, classes)

    #  Print some details about our data set so that we can make sure it looks good
    print_dataset_details(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y)

    #  Convert each image into a list of pixels instead of a 3-Dimensional Array
    train_set_x_flatten, test_set_x_flatten = unroll_images(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y)

    #  Normalize our images for each dataset, this means to make them all a similar scale/size
    train_set_x = normalize_dataset(train_set_x_flatten)
    test_set_x = normalize_dataset(test_set_x_flatten)

    # Create our model (neuron) and train it
    trained_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    #  Determine the height/width of our pictures
    num_px = train_set_x_orig.shape[1]

    #  Show a random picture that the model was wrong and misclassified
    show_incorrect_classification(trained_model, num_px, test_set_x, test_set_y, classes)

    #  Plot learning curve (with costs)
    #  This shows a graph of the cost over time as the model optimizes
    plot_learning_curve(trained_model)

    #  Plot the cost function over time for various choices for learning rates
    plot_effect_of_learning_rate(train_set_x, train_set_y, test_set_x, test_set_y)
