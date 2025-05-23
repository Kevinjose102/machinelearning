from datetime import date
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# loading the data
X,Y = load_coffee_data();
print(X.shape, Y.shape)

# normalize the data 
norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X)
Xn = norm_l(X)

# tile the date
Xt = np.tile(Xn, (1000,1))
Yt = np.tile(Y, (1000,1))

tf.random.set_seed(1234)

model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(units = 3, activation = 'sigmoid', name = "layer_1"),
    Dense(units = 1, activation = 'sigmoid', name = "layer_2")
])


# setting the weights for each layer
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])


model.get_layer("layer_1").set_weights([W1,b1])
model.get_layer("layer_2").set_weights([W2,b2])

model.summary()

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# epchos means how many times should the training set be applied to the model
model.fit(
    Xt,Yt,            
    epochs=10,
)