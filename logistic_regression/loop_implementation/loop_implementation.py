import numpy as np
import math


"""
 A pseudo code implementation of gradient descent to help understand the concepts and process.
 Don't run this.
"""

# Define our sigmoid function for later use
# Remember the sigmoid just constrains our models prediction to be a value between 0 and 1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Here we define our alpha term, commonly referred to as the learning rate in practice.
# This is a constant value that determines how big of a move to make for a given slope update.
learning_rate = .5

# Fictitious input values (images) matrix, for this psuedo example lets consider this a matrix where each i value is
# actually an image in matrix form
x = []

# Fictitious matrix representing all of the labels for our training images
# this matrix defines if there is a cat or not in a specific image held within x
y = []

# Current b and weight values
b = 0
w = []

# Cost Function
J = 0

# The slope of our weights and b value for a specific image
dw1 = 0
dw2 = 0
db = 0

# The amount of images we have in our training set
m = 200

# Iterate over each image
for i in range(m):
    # Logistic Model Equation to make our prediction, z = w transpose x + b
    # w is a matrix of constant values representing the current weights (randomly initialized at some previous point)
    # x is a matrix representing our image and b is a constant number
    z = np.transpose(w, x) + b

    # Compute our actual prediction by constricting our models output
    # to be a value between 0 and 1 which we can interpret as cat or no cat
    # Ex. Greater than .5 will be defined as a cat and less than .5 not a cat
    a = sigmoid(z)

    # Compute the cost function relative to this prediction to see how accurate we were
    # We keep a running total of the cost across all images since we are iterating over
    # all images in this loop defined by m
    J += -((y[i] * np.log(a)) + (1 - y[i]) * np.log(1 - a))

    # Compute the derivative value of loss with respect to z.
    # That is find the slope of our loss function relative to our sigmoid function
    # Which is also the same as saying take our predicted value for this loop iteration and
    # find where we are on the loss curve and find the slope at that point.
    # This slope is important because we can use it to determine how to progress downward to minimize the loss
    dz = a - y[i]

    # NOTE THIS SHOULD BE A FOR-LOOP OVER ALL VALUES OF X
    # BUT FOR THE SAKE OF SIMPLICITY WE HAVE ONLY SHOWN TWO OF THE VALUES
    # FOR THE SAKE OF CLARITY OF WHATS HAPPENING
    # Computing the derivatives of all of the input feature weights, one for each feature, approximately 50,000
    dw1 += x[0] * dz
    dw2 += x[1] * dz
    # THIS WOULD BE THE END OF THE FOR-LOOP

    # Calculate the slope value of our loss function relative to b
    db += dz


# Since this computed all of the values related to our loss function we need to average these terms to
# bring them in alignment with our cost function definition.
# Remember the cost function is the averaged sum of all the loss across all of our images

J /= m
dw1 /= m
dw2 /= m
db /= m

# Remember the dw1 and dw2 terms are for example purposes only, there would be about 50,000 of these values
# and dividing by m is simply averaging these terms across the amount of images we developed them over


"""
 Finally, we have completed a single iteration over all of our training images.
 This means the same thing as saying we completed a single iteration of gradient descent.

 Since we have done a single iteration of gradient descent we can now perform a small update (adjustment/optimization)
 to all of our w values (weights) and b. Then we can go back to this for-loop thousands of times until we get better
 and better values of our constant terms w's (weights) and b.
"""

w[0] -= learning_rate * dw1
w[1] -= learning_rate * dw2
b -= learning_rate * db
