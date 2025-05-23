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

# FOR LOGISTIC REGRESSION now

# the data set intialization
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

# returns a boolean array 
pos = Y_train == 1
neg = Y_train == 0

model = Sequential(
    [
        Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

# returns the architecture of the network in a table format
model.summary()

# setting the weights and bias for the first layer
logistic_layer = model.get_layer('L1')
set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

# predicting using logistic in tensorflow
a1 = model.predict(X_train[0].reshape(1,1))
print(a1)

plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)