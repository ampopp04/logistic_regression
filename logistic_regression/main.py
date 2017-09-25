from logistic_regression.utils.data_utils import load_dataset
from logistic_regression.utils.data_utils import unroll_images
from logistic_regression.utils.data_utils import normalize_dataset

from logistic_regression.utils.model_utils import run_trained_model_on_image


"""
The below code loads labeled training and test data sets of images of cats and not cats. It uses these images
to create and train a single neuron via supervised learning through gradient descent.

A simple neuron activation function of logistic regression is used.

Even with such a simple neuron model (one neuron) and a limited size data set we are able to produce
a neuron with accuracy in excess of 70% on new images.
"""

# Load image data sets along with their correct label value data sets (Contains cat or not)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#  Convert each image into a list of pixels instead of a 3-Dimensional Array of pixels
train_set_x_flatten, test_set_x_flatten = unroll_images(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y)

#  Normalize our images, this means to make them all a similar scale/size
train_set_x = normalize_dataset(train_set_x_flatten)
test_set_x = normalize_dataset(test_set_x_flatten)

#  Determine the height/width of our pictures
num_px = train_set_x_orig.shape[1]

# Image name from data_sets directory to run prediction model against once it is fully trained
image_name = "penny-jump.jpg"

#  Create and train our neuron and then run it against this image to predict if there is a cat in it or not
#  This method has additional default parameters you can adjust to see how they change the results
run_trained_model_on_image(image_name, train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes)


"""
The above code is all that is needed to test your own cat image against the model.
The model (neuron) will train itself, using training data, prior to making a prediction on your new image.
You can dig into the various methods to gain a deeper understanding for how it works.

Below is a method that will create and train the model while providing many console outputs
and graphs detailing various parts of the creation and training process.  This might be a useful
place to start if you want to dig into how the code works.

Keep in mind that whenever it plots an image it will halt the code process until you close that image.
Then it will move onto the next output or image/graph.
"""

# NOTE: UNCOMMENT THE FOLLOWING LINE OF CODE TO SHOW VARIOUS DETAILS OF THE TRAINING PROCESS
# show_all_details_of_learning_and_optimization_process(train_set_x_orig, test_set_x_orig, train_set_y, test_set_y, classes)