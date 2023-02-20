#!/usr/bin/env python
# coding: utf-8

# # Advanced numpy
# 
# `For-loops are forbidden here!!`
# 
# `a) Give a code snippet that create two random 1D arrays of length 20, with`
# `integer entries between 0 and 3 (included). Then, the code should compute the`
# `confusion matrix from these two vectors.`

import numpy as np
import pandas as pd

arr1 = np.random.randint(0,4,20)
arr2 = np.random.randint(0,4,20)

confusion_matrix = np.zeros((4,4), dtype= int)
np.add.at(confusion_matrix, (arr1, arr2), 1)
print(confusion_matrix)

# `Give a code snippet that generates two 1D arrays with values from 0 to 19`
# `(included) in ascending order. Then, the code should shuffle both arrays such`
# `that same positions contain same values after shuffling (like you would shuffle`
# `train data and labels).`

arr1 = np.arange(0,20)
arr2 = np.arange(0,20)

print("Arrays before shuffling:")
print(arr1)
print(arr2)

shuffled_indices = np.random.randint(0,20,20)
arr3 = arr1[shuffled_indices]
arr4 = arr2[shuffled_indices]

print("Arrays after shuffling:")
print(arr3)
print(arr4)

# `c) Give a code snippet that creates a 1D array with random values from 0 to`
# `9 (included). Then, interpret this array as scalar targets and create a one-hot`
# `representation for them, assuming 10 classes.`

arr1 = np.arange(0,10)
indices = np.arange(10)

one_hot = np.zeros((10,10), dtype= int)
one_hot[arr1, indices] = 1

print(one_hot)


# # CNN 2 answer
# 
# `model = Sequential([`
# 
#     Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)),
# 
#     MaxPooling2D(pool_size=(2, 2)),
# 
#     Conv2D(filtersv = 64, kernel_size = (3, 3), padding='same', activation='relu'),
# 
#     MaxPooling2D(pool_size=(2, 2)),
# 
#     Flatten()
# 
#     Dense(units = 128, activation='relu'),
# 
#     Dense(units = 64, activation='relu'),
# 
#     Dense(units = 10, activation='softmax')
#     ])

# # True & False
# 
# 1. All filters in a Conv2D-layer must be identical: `False`. In a Conv2D-layer, each filter has its own set of weights, which means that the filters can have different shapes and sizes.
# 
# 2. The number of filters in a conv2D-layer can be chosen arbitrarily: `True`. The number of filters is a hyperparameter that can be set when designing the model architecture.
# 
# 3. The width and height of a conv2D-layer output can be chosen arbitrarily: `True`. The width and height of the output of a Conv2D-layer depend on the input size, the padding, the stride, and the size of the filter. These parameters can be chosen to achieve the desired output size.
# 
# 4. A max-pooling layer must have a kernel size of 2x2: `False`. The kernel size for max-pooling can be chosen arbitrarily, but a kernel size of 2x2 is a common choice.
# 
# 5. A Conv2D-layer must be followed by ReLU: `False`. While ReLU is a common activation function used after a Conv2D-layer, other activation functions can also be used, or no activation function at all.
# 
# 6. A CNN classifier must contain at least one affine layer: `True`. An affine layer, also called a fully connected layer, is usually used as the final layer in a CNN classifier to produce the output.
# 
# 7. A CNN classifier can contain an arbitrary number of Conv2D-layers: `True`. The number of Conv2D-layers in a CNN classifier can be chosen based on the complexity of the problem being solved and the available computational resources.

# # Last year True & False
# • A Conv2D layer is a special case of a affine layer - `False`. Conv2D layer applies a convolution operation to the input tensor, whereas affine layer applies a matrix multiplication operation.
# 
# • An affine layer must always be followed by a Softmax layer - `False`. An affine layer can be followed by various activation functions, depending on the task and network architecture.
# 
# • A max-pooling layer has no trainable parameters - `True`. A max-pooling layer performs a fixed operation (taking the maximum value) and has no parameters to learn during training.
# 
# • A Conv2D-layer must be followed by MaxPooling2D - `False`. While it is common to use a MaxPooling2D layer after a Conv2D layer to reduce the spatial dimensions, it is not required.
# 
# • A CNN/DNN classifier must contain at least one Conv2D layer - `False`. While Conv2D layers are commonly used in CNNs, other layers like Dense layers can also be used to build a classifier.
