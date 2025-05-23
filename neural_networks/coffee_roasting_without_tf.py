import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# loading the data 
X,Y = load_coffee_data();
print(X.shape, Y.shape)

# plotting the data 
plt_roast(X,Y)

# normalizing the data 
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

g = sigmoid


# okay now instead of hardcoding all the neurons in the network 
# define functions for the types of layers (like dense)
def dense(a_in, W, b):
    units = W.shape[1] # the number of neurons depends on the number of output activations 
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j] # returns the jth column of W
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return(a_out)

# now to defin the model 
# the activation of the last layer should be passed as the input to the next layer
def Sequential(x, W1, b1, W2, b2):
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    return a2

# the trained weights and biases from the previous one done using Tensorflow.

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

# now to predict 
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = Sequential(X[i], W1, b1, W2, b2)
    return(p)


X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # always normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")



# this part is for predicting if a given image is a 1 or a 0
# same concept tho

# YOU CAN ALSO VECTORIZE THE ENTIRE NETWORK 
# making it was more efficient w
# UNQ_C3
# UNGRADED FUNCTION: my_dense_v

def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    Z = np.matmul(A_in, W) + b # matmul is basially .dot() for matrices
    A_out = g(Z)
    return(A_out)

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return(A3)

