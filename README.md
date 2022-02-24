## Neural-Networks-Image-Recognition

# MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python

Third Project - Neural Network Classifier for Image Recognition

The third project for the MIT MicroMasters Program course on Machine Learning with Python dealt with
Neural Networks in order to build a network to classify images of single and double digits using the famous MNIST (Mixed National
Institute of Standards and Technology) database.

The MNIST database contains binary images of handwritten digits collected from among Census Bureau employees and high school students, and it is commonly used to train image processing systems. The database contains 60,000 training images and 10,000 testing images; All of which have been size-normalized and centered in a fixed size of 28 x 28 pixels.

The project started by first creating a neural network from scrath, and then moving towards single digit recognition by implementing more sophisticated deep neural networks with PyTorch, and convolutional networks with some experimenting with layer design and hyperparameter tuning, before finishing off with double digit
recognition. 

Additional helper functions were given to complete the project in two weeks of time.

## Dataset

The function call get_MNIST_data() returns the following Numpy arrays:

- train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are
the raw pixel values flattened out into a vector of length 28^2 = 784. The pixel values are float values
between 0 and 1 (0 for black, 1 for white, and various shades of gray in-between.

- train_y : The labels for each training datapoint that are the digit numbers for each image (i.e. a number between 0-9).

- test_x : A matrix of the test data, formatted the same way as the training data.

- test_y : The labels for the test data, formatted the same wat as the training data.

## Access and requirements

The different models are run each separetely from their files with the help of the three modules (train_utils.py, utils.py and utils_multiMNIST.py) that contain helper and utility functions. Neural_nets.py contains the basic model for the neural network while the other files - conv.py, mpl.py, nnet_cnn.py, and nnet_fc.py -  use PyTorch libraries.

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt
