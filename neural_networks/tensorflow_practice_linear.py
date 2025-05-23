import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy # type: ignore
from tensorflow.keras.activations import sigmoid # type: ignore
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic

# this part is to suppress the tensorflow logs for a cleaner output
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)


# this creates a SINGLE NEURON
# linear activation means that its JUST w.x + b and no other transformation is done
# this only DECLARES the layer hence no weight are added here
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )

# to MANUALLY set the weights and bias
set_w = np.array([[200]])
set_b = np.array([100])

linear_layer(X_train)
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

# this should mean give the output of w.x + b 
# where x = X_train[0] which is 1
    #   w = 200
    #   b = 100
# a1 -> activation value 
# should return 300
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

prediction_tf = linear_layer(X_train) # USING TENSORFLOW
prediction_np = np.dot(X_train, set_w) + set_b # USING NUMPY
# both give the same thing

plt_linear(X_train, Y_train, prediction_tf, prediction_np)
