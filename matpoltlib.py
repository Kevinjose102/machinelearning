import numpy as np
import matplotlib.pyplot as plt

def plot_one_variable(x, y):
    """
    Scatter plot for one feature (x) vs output (y)
    Red X = y=1, Blue O = y=0
    """
    plt.figure(figsize=(6,4))
    plt.scatter(x[y==1], y[y==1], color='red', marker='x', label='y=1', s=100)
    plt.scatter(x[y==0], y[y==0], color='blue', marker='o', label='y=0', s=100)
    plt.title('one variable plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_two_variable(X, y):
    """
    Scatter plot for two features X[:,0], X[:,1] colored by y.
    Red X = y=1, Blue O = y=0
    """
    plt.figure(figsize=(6,4))
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='x', label='y=1', s=100)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='o', label='y=0', s=100)
    plt.title('two variable plot')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, y, theta):
    """
    Plots linear decision boundary for two variables.
    theta is parameter vector [theta0, theta1, theta2] for line: theta0 + theta1*x + theta2*y = 0
    """
    plt.figure(figsize=(6,4))
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='x', label='y=1', s=100)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='o', label='y=0', s=100)
    
    # Decision boundary line: x1 = -(theta0 + theta1 * x0) / theta2
    x_vals = np.array([np.min(X[:,0]) - 1, np.max(X[:,0]) + 1])
    y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]
    
    plt.plot(x_vals, y_vals, color='blue', linewidth=3)
    plt.fill_between(x_vals, y_vals, y2=np.min(X[:,1]) - 1, color='lightblue', alpha=0.5)
    
    plt.title('Decision boundary')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.grid(True)
    plt.xlim(np.min(X[:,0]) - 1, np.max(X[:,0]) + 1)
    plt.ylim(np.min(X[:,1]) - 1, np.max(X[:,1]) + 1)
    plt.show()

def plot_sigmoid():
    """
    Plot the sigmoid function with shaded areas for z<0 and z>=0 and annotations.
    """
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(6,4))
    plt.plot(z, sigmoid, linewidth=5, color='blue')
    plt.title('Sigmoid function')
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    
    # Shade regions
    plt.fill_between(z, 0, 1, where=(z < 0), color='lightblue', alpha=0.5)
    plt.fill_between(z, 0, 1, where=(z >= 0), color='lightcoral', alpha=0.5)
    
    # Annotations
    plt.annotate('z < 0', xy=(-3, 0.5), xytext=(-7, 0.7),
                 arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10, color='blue')
    plt.annotate('z >= 0', xy=(3, 0.5), xytext=(2, 0.7),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')
    
    plt.grid(True)
    plt.show()

# -----------------------
# Example usage of these functions:

# Example data for one variable plot
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1, 1])
plot_one_variable(x, y)

# Example data for two variable plot
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [2, 2], [3, 0.5]])
y = np.array([0, 0, 0, 1, 1])
plot_two_variable(X, y)

# Example theta for decision boundary
theta = np.array([-3, 1, 1])  # corresponds roughly to line x1 = -(-3 + 1*x0)/1

plot_decision_boundary(X, y, theta)

# Plot sigmoid function
plot_sigmoid()
