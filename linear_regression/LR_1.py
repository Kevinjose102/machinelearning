import numpy as np
import matplotlib.pyplot as plt

# the given data for supervised learning
x_train = np.array([1.0, 2.0]) # the input x 
y_train = np.array([300.0, 500.0]) # the output y
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
m = x_train.shape[0]
# or also len(x_train) works

# to get a particular data from the training set use the indexing method as normal arryas
i = 0 # the index you want to access
x_i = x_train[i]

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# c is for colour
# marker is the way you want to display the mark in the graph
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 200
b = 100

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
      w is the weight 
      b is the bias
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


# after finding the right weight and bias we can predict 
# with any x value a corresponding y vlaue

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")