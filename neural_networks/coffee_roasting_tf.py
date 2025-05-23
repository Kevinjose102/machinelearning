import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# HELPER FUNCTIONS FROM DEEPLEARNING.AI
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# THE MODEL CREATED WILL NOT BE TRAINED HERE 
# THIS IS ONLY TO USE THE MODEL WITH THE PRE-DEFINED WEIGHTS AND BIASES 
# TO PREDCIT AND SEE IF IT STILL WORKS WITHOUT TF

# loading the data
X,Y = load_coffee_data();
print(X.shape, Y.shape)

plt_roast(X, Y)

# normalize the data 
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1) # creates a normalization layer, 
# axis = -1 means each column -> feature
norm_l.adapt(X)  # calculates mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# Tile/copy our data to increase the training set size and reduce the number of training epochs.
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)

# Sets random seed for reproducible model results (e.g. same weight init each run)
tf.random.set_seed(1234)

model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(units=3, activation="sigmoid", name="layer_1"),
    Dense(units=1, activation="sigmoid", name="layer_2")
])

# setting the weights and bias
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])

# Replace the weights from your trained model with
# the values above.
model.get_layer("layer_1").set_weights([W1,b1])
model.get_layer("layer_2").set_weights([W2,b2])

model.summary()

print("BEFORE")
W1, b1 = model.get_layer("layer_1").get_weights()
W2, b2 = model.get_layer("layer_2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# epchos means how many times should the training set be applied to the model
model.fit(
    Xt,Yt,            
    epochs=10,
)
# the output will have like 6250 number in total
# that is not the total number of training data BUT 
# the data that is divided into batches
# the actual data is way more (in 200000)

# after fitting the data the weights and biases are :
print("AFTER")
W1, b1 = model.get_layer("layer_1").get_weights()
W2, b2 = model.get_layer("layer_2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)


# NOW TO PREDICT SOMETHING BASED ON THE TRAINED MODEL
x_test = np.array([
    [200,13.9],
    [200,17]
])
# you have to always normalize the data 
X_testn = norm_l(x_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

# the predictions will return a value between 0 and 1 
# signifying the probability of the test data being a "good" coffee
# so BASICALLY if greater than 0.5 it should mean that it is good coffee
# else BAAAD 
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")