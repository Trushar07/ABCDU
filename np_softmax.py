import numpy as np

def softmax(x):

    exp_x = np.exp(x)
    row_exp_sum = np.sum(np.exp(x), axis = 1) #for 1D array axis = 0
    row_exp_sum_reshaped = row_exp_sum.reshape(-1,1)
    # The -1 argument in the tuple is a placeholder, which tells the method to 
    # automatically determine the appropriate size of this dimension based on the 
    # size of the other dimensions and the total number of elements in the array.
    return exp_x / row_exp_sum_reshaped

x = np.array([[-1, -1, 5], [1, 1, 2]])

result = softmax(x)

print(result)