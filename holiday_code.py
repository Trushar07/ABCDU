import numpy as np ;
import requests ;
import os ;
import sys ;
import matplotlib.pyplot as plt

# MNIST

if os.path.exists("./mnist.npz") == False:
    print ("Downloading MNIST...") ;
    fname = 'mnist.npz'
    url = 'http://www.gepperth.net/alexander/downloads/'
    r = requests.get(url+fname)
    open(fname , 'wb').write(r.content)

## read it into 'traind' and 'trainl'
data = np.load("mnist.npz")
traind = data["arr_0"] ;
trainl = data["arr_2"] ;


# 1. Python list creation and list comprehension

ex1_1 = [num for num in range(10,21) if num % 2 == 1]
print(ex1_1)

ex1_2 = [num for num in range(100, -1, -10) if num % 10 == 0]
print(ex1_2)

ex1_3 = [num for num in range(15, 0, -1) if num % 3 == 0]
print(ex1_3)

for i in range(1,11):
    temp = "x" * i
    print(temp)

for i in range(5, 0, -1):
    temp = "string" + str(i)
    print(temp)

ex1_6 = ["1", 1 , 1.0, "one"]
print(ex1_6)

ex1_7 = [num for num in range(100) if str(num).find("5") != -1]
print(ex1_7)


# 2. Array creation

ex2_1 = np.arange(-100, 1, 2)
print(ex2_1)

ex2_2 = np.array([[1,1], [2,2],[3,3]])
print(ex2_2)

# ex2_2 = np.ones((3,2))
# for i in range(3):
#   ex2_2[i] *= (i + 1)
# print(ex2_2)

ex2_3 = np.ones((3,2))
ex2_3 *= -1
print(ex2_3)

ex2_4 = np.random.normal(loc= 0, scale= 1, size=(5,4,3))
# loc is mean and scale is standard deviation.
print(ex2_4)


# 3. Numpy basics and slicing

td = np.zeros((50,5,5))

for i in range(50):
    td[i, :, :] = i

print(td)

x = td[0]
print(x)

td[9, -2:, :] = -1
print(td[9])

print(np.mean(td[9]))

x1 = td[9, ::3, :]
print(x1)

x2 = td[9, :, ::3]
print(x2)

x3 = td[9, ::-1, :]
print(x3)

x4 = x3[::2, :]
print(x4)

td *= 1
td += 1
print(td[9])


# 4. Reduction

td = np.zeros((50,5,5))

for i in range(50):
    td[i, :, :] = i

pixel_var = np.var(td[:, 0, 0])
print("Pixel variance for pixel 0,0 over all samples:", pixel_var)

pixel_argmax = np.argmax(td[:, 0, 0])
print("Pixel argmax for pixel 0,0 over all samples:", pixel_argmax)

std_image = np.std(td, axis=0)
print("The “standard deviation image” over all samples:\n", std_image)

row_wise_mean = np.mean(td, axis=1)
print("The row-wise mean over all samples:\n", row_wise_mean)

column_wise_mean = np.mean(td, axis=2)
print("The column-wise mean over all samples:\n", column_wise_mean)


# 5. Broadcasting
# 
# Using ‘td’ ...
# 
# a) create a 5-element row vector with entries from 1 to 5, and subtract it from
# all rows of all samples using broadcasting
# 
# b) create a 5-element column vector with entries from 1 to 5, and multiply it
# with all columns of all samples using broadcasting
# 
# c) compute the mean image over all samples, and subtract it from all samples
# via broadcasting

td = np.zeros((50,5,5))

for i in range(50):
    td[i, :, :] = i

row_vector = np.array([1,2,3,4,5])
print(td - row_vector)

column_vector = row_vector.reshape(5,1)
print(td * column_vector)

mean = np.mean(td, axis= 0)
print(td - mean)


# 6 Fancy indexing and mask indexing
# 
# a) create a 20-element vector with entries from 1 to 20, and copy out all
# elements that are even using mask indexing!
# 
# b) create a 20-element vector with entries from 1 to 20, and copy out elements
# at positions 1, 5 amd 10 using a single operation!

vec = np.arange(1,21)
mask = (vec % 2 == 0)
even_vec = vec[mask]
print(even_vec)

indices = [1,5,10]
new_vec = vec[indices]
print(new_vec)


# 7 Matplotlib.
# 
# a) plot the function 1/x between 1 and 5 using 100 support points!
# 
# b) generate a scatter plot of the same data as in a)!
# 
# c) generate a bar plot of the same data as in a)!
# 
# d) plot 1/x and √x together in a single plot, same range as before
# 
# e) generate 100 numbers distributed according to a uniform distribution between 0 and 1, and display their histogram!

x = np.linspace(1, 5, 100)
y = 1 / x

fig, ax = plt.subplots(1,1)
ax.plot(x,y)
plt.show()


x = np.linspace(1, 5, 100)
y = 1 / x

fig, ax = plt.subplots(1,1)
ax.scatter(x,y)
plt.show()


x = np.linspace(1, 5, 100)
y = 1 / x

fig, ax = plt.subplots(1,1)
ax.bar(x,y)
plt.show()


x = np.linspace(1, 5, 100)
y = 1 / x
z = np.sqrt(x)

fig, ax = plt.subplots(1,1)
ax.plot(x,y)
ax.plot(x,z)
plt.show()


x = np.random.uniform(0, 1, 100)

plt.hist(x, rwidth=0.8)
plt.show


# 8 MNIST and matplotlib
# 
# Now use MNIST data (pre-loaded in the code file) where ’traind’ contains the
# samples, ’trainl’ the target values (labels) in one-hot format.
# 
# a) Display samples nr. 5,6 and 7 in a single figure!
# 
# b) Compute the mean pixel value for each image and display all means in a
# scatter plot!
# 
# c) Copy out all the images whose mean pixel value is > 0.3 and display 3 of
# them
# 
# d) Compute the “variance image” over all samples and display it!
# 
# e) Copy out 10 random images and display them in a single figure! 
# 
# f) Copy out all samples of class 5 and display 10 of them!

indices = [5,6,7]

fig, ax = plt.subplots(1,3)

for i, index in enumerate(indices):
    ax[i].imshow(traind[index])

plt.show()


mean_pixel_value = np.mean(traind, axis = (1,2))

x = np.linspace(0, 60000, 60000)

plt.scatter(x, mean_pixel_value)
plt.show()


mask = mean_pixel_value > 0.3

images = traind[mask]

fig, ax = plt.subplots(1,3)

for i in range(3):
    ax[i].imshow(images[i])

plt.show()


variance = np.var(traind, axis= (1,2))
x = np.linspace(0, 60000, 60000)

plt.plot(x, variance)
plt.show()

indices = []

for i in range(10):
    indices.append(np.random.randint(0,60000))

fig, ax = plt.subplots(1,10)

for i, index in enumerate(indices):
    ax[i].imshow(traind[index])

plt.show()


numerical_classes = trainl.argmax(axis=1)
print(numerical_classes.shape)
mask = (numerical_classes == 5)
class_5_samples = traind[mask]

fig, ax = plt.subplots(1,10)

for i in range(10):
    ax[i].imshow(class_5_samples[i])
plt.show()


# 11 Implementing softmax in TF
# 
# Write a python function S(X) which takes an 2D TF tensor and returns the
# softmax, applied row-wise, as a TF tensor. The function must work for tensors
# X with an arbitrary number of rows! Print out results for ~x = [−1, −1, 5] and
# ~x = [1, 1, 2]!

import tensorflow as tf

def softmax(x):
    e = tf.exp(x)
    row_wise_sum = tf.reduce_sum(e, axis = 0, keepdims= True) 
    return e / row_wise_sum

x1 = tf.constant([-1, -1, 5], dtype=tf.float32)
x2 = tf.constant([1, 1, 2], dtype=tf.float32)
x3 = tf.constant([[-1, -1, 5], [1, 1, 2]], dtype=tf.float32)

print("Softmax for x1:\n", softmax(x1))
print()
print("Softmax for x2:\n", softmax(x2))
# print()
# print("Softmax for x3:\n", softmax(x3))
""" Not working for arbitrary number of rows"""


# # Implementing cross-entropy in TF
# `Write a python function MSE(Y,T) which takes an 2D TF tensor and returns the its mean-squared-error`
# `(MSE) loss as a TF scalar!`

def MSE(Y, T):
    return tf.reduce_mean(tf.square(Y - T))

Y = tf.constant([[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.8, 0.1, 0.1]])
T = tf.constant([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype= float)
print(MSE(Y, T))
print(MSE(Y[0:2], T[0:2]))


# # 20 TF Gradients

def f(x):
      tmp = tf.range(1, x.shape[0] + 1, 1.0) ; # create 1D vector that counts up from 1
      return tf.reduce_sum(x * tmp) ;

a1 = tf.constant([1., 2, 3]) ; # decimal point to have float dtype
a2 = tf.constant([2., 0, 2]) ; # for a1 and a2
print ("20a", f(a1), f(a2)) ;

with tf.GradientTape(persistent=True) as g: # persistent = True if g is used for several gradient computations, not just one
    g.watch(a1) ;  # need to watch both constants, otherwise grad w.r.t. both will be None
    g.watch(a2) ;  # This is because by default, TF watches only tf.Variables
    y1 = f(a1) ;
    y2 = f(a2) ;

print ("20b", g.gradient(y1,a1)) ;
print ("20c", g.gradient(y2,a2)[0]) ; # TF always computes whole gradient, from which we extract entry nr. 1

