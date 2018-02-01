"""
The initial building block of Keras is a model, and the simplest model is called sequential. A
sequential Keras model is a linear pipeline (a stack) of neural networks layers. This code fragment
defines a single layer with 12 artificial neurons, and it expects 8 input variables (also known as
features):
"""

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))

"""
Each neuron can be initialized with specific weights. Keras provides a few choices, the most
common of which are listed as follows:

random_uniform: Weights are initialized to uniformly random small values in (-0.05, 0.05). In other
words, any value within the given interval is equally likely to be drawn.

random_normal: Weights are initialized according to a Gaussian, with a zero mean and small
standard deviation of 0.05. For those of you who are not familiar with a Gaussian, think about a
symmetric bell curve shape.

zero: All weights are initialized to zero.
"""
