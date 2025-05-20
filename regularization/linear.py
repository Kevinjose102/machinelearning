import numpy as np
import matplotlib.pyplot as plt

def compute_cost_linear_reg(x, y, w, b, lambda_ = 1):
    m = x.shape[0]
    n = len(w)
    cost = 0

    for i in range(m):
        f_wb_i = (np.dot(w, x[i]) + b)
        cost += (f_wb_i - y[i]) ** 2
    
    cost = cost / (2 * m)

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j] ** 2)

    reg_cost = (reg_cost * lambda_) / (2 * m)

    total_cost = cost + reg_cost

    return total_cost

def compute_gradient_linear_reg(x, y, w, b, lambda_):
    m,n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = (np.dot(w, x[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    
    dj_db /= m
    dj_dw /= m


    for i in range(n):
        dj_dw[i] += (lambda_ / m) * w[i]

    return dj_dw, dj_db


