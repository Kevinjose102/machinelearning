import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')

# the function for using exp in the numpy module
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

def sigmoid(z):

    g = 1 / (1 + np.exp(-z))
    
    return g

# the z that is passed will be the f(x,b) based from regression
