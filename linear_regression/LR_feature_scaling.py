import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
import copy

# load the dataset
x_train, y_train = load_house_data()
x_features = ['size(sqft)','bedrooms','floors','age']


def compute_cost(x, y, w, b):
    cost = 0
    m = x.shape[0]

    for i in range(m):
        f_wb_i = np.dot(w, x[i]) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)

    return cost

def compute_gradient(x, y, w, b):
    m, n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = (np.dot(w, x[i]) + b ) - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
    J_history = []

    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):

        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(cost_function(x, y, w, b))

    return w, b, J_history

# okay now for finding a "good" learning rate alpha
# using z-score normalization 

def zscore_normalize_features(x):
    # calculate MEAN for each column / feature
    mu = np.mean(x, axis = 0) # will be a one dimensional array
    # same for std deviation
    sigma = np.std(x, axis = 0)

    # the normalized features
    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma

# okay now instead of running the gradient descent on the base x_train data 
# we run the thing with the normalized features

x_norm, x_mu, x_sigma = zscore_normalize_features(x_train)

print(f"X_mu = {x_mu}, \nX_sigma = {x_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

w_init = np.zeros(x_train.shape[1])
b_init = 0
alpha = 1.0e-1
w_norm, b_norm, hist = gradient_descent(x_norm, y_train, w_init, b_init, compute_cost, compute_gradient, alpha, 1000)

print(f"w : {w_norm}")
print(f"b : {b_norm}")