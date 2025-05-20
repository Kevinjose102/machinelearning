import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic_reg(x, y, w, b, lambda_ = 1):
    m = x.shape[0]
    n = len(w)
    cost = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, x[i]) + b)
        cost += (-y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i))
    
    cost = cost / m

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j] ** 2)

    reg_cost = (reg_cost * lambda_) / (2 * m)

    total_cost = cost + reg_cost

    return total_cost

def compute_gradient_logistic_reg(x, y, w, b, lambda_):
    m,n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = sigmoid(np.dot(w, x[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    
    dj_db /= m
    dj_dw /= m


    for i in range(n):
        dj_dw[i] += (lambda_ / m) * w[i]

    return dj_dw, dj_db


np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )