import numpy as np
import matplotlib.pyplot as plt
import copy, math

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))

    return g

def compute_cost_logistic(x, y, w, b):
    cost = 0.0

    m = x.shape[0]

    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)

        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost / m
    return cost


# now to calculate the gradient of the cost funciton J 
def comput_gradient_logistic(x, y, w, b):
    m, n = x.shape
    
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, x[i]) + b)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += (error * x[i, j])
        dj_db += error
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def compute_gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = comput_gradient_logistic(x, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        if i<100000:      # prevent resource exhaustion FOR WHATEVER REASON
            J_history.append(compute_cost_logistic(x, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history

w_tmp  = np.zeros_like(x_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = compute_gradient_descent(x_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")