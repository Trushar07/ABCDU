import numpy as np

def f(x):
    return (x[0]**2) + (2*x[1]**2)

def df(x):
    return np.array([2 * x[0], 4 * x[1]])

x = np.array(input("Enter the starting points separated by comma(,):").split(","), dtype= int)

learning_rate = float(input("Enter the step size:"))

number_of_iterations = int(input("Enter number of Iterations:"))

for i in range(number_of_iterations):

    grad = df(x)

    x = x - learning_rate * grad

    print(f"Iteration {i}: x = {x} and f(x) = {f(x)}")