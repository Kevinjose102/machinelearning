import numpy as np
import matplotlib.pyplot as plt
import copy

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# this means that there are 4 types of features 
# as x1 x2 x3 and x4
# hence 4 weights also needed

# random values
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# using the cool "vectorization" andrew mentioned
# and we can make a prediction

# will return the cost at a particular weight and bias
# w and b will change according to the gradient fuction
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

cost = compute_cost(x_train, y_train, w_init, b_init)
print(f"Cost at optimal w : {cost}")

# for calculating the gradient that is dJ/dw and dJ/db
# but as there are n number of features there is a double loop required
def compute_gradient(x, y, w, b):
    m,n = x.shape

    # as there are n weights
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        # as there are m number of data
        error = (np.dot(x[i], w) + b) - y[i]
        # as there are n features for each data entry
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# now putting everything togther
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(cost_function(x, y, w, b))

    return w, b, J_history



initial_w = np.zeros_like(w_init)
initial_b = 0

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()

