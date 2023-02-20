import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.random.uniform(-1, 1, size=(3,3))
print("ReLU output:\n",relu(x))

y = np.linspace(-5, 5, 100)
z = relu(y)

plt.plot(y,z)
plt.title("ReLU function")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()