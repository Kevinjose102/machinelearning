import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
np.set_printoptions(precision=3, suppress=True)


# creating the data set
# the data here will be generated in like groups 
# like each class will be around a center in this case 
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
print(X_train[0])
print(y_train) # will have the values as 0 1 2 3 as there are 4 groups 

# JUST TO VISUALIZE THE DATA
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=30, alpha=0.6, edgecolors='k')
plt.title("Generated Blobs Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001), # supposed to be the better version of the gradient descent 
    # havent learnt this till now but okay 
    # trusting the process :prayemoji
)

model.fit(
    X_train,y_train,
    epochs=10 # why only 10 tho HMMMMMMMM
)

p_nonpreferred = model.predict(X_train)
print(p_nonpreferred[:10])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))


# THE ABOVE METHOD IS THE MORE DIRECT AND GENERIC ONE 
# FOR MAKING THIS BETTER DO THIS 

preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear') # USING LINEAR INSTEAD OF SOFTMAX
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # rearranges the terms so that there is VERY LESS numerical error 
    # the outputs in this form are referred to as *logits*.
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)

p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")

# but the output here for will not be the probablities like last time 
# this output is should be further sent to a softmax function to calculate the P()
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")

# to print which is the most likely category for that 
for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")


        