import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":

    if sys.argv[1] == "2.1":

        X = np.random.uniform(0 ,1, (10, 5, 3, 3))

        Y = X[2]
        print(Y)

        X[0, -3::, :, :] = -1
        print(X[0])

        W = X[3, ::2, :, :]
        print("W: ", W)

        Z = X[[0,1,3]]
        print("Z:", Z)

        sum = np.sum(X, axis = (1,2,3))
        mask = sum < 40
        M = X[mask]
        print(M)

    if sys.argv[1] == "2.2":
        """ Advanced Numpy """

        x = np.arange(0,20)
        y = np.arange(0,20)

        print("x and y before shuffling")
        print("x:",x)
        print("y:",y)

        indices = np.random.randint(0,20,20)

        x = x[indices]
        y = y[indices]

        print("x and y after shuffling")
        print("x:",x)
        print("y:",y)

    if sys.argv[1] == "2.3":
        """ Matplotlib """

        x = np.linspace(0,2, 100)
        y = x**3

        fig, ax = plt.subplots(1,1)
        plt.plot(x,y)
        plt.show()

        images = np.random.uniform(0,1, (10,20,20))

        fig, ax = plt.subplots(1,10)
        for i in range(10):
            ax[i].imshow(images[i])
        plt.show()

        z = np.random.uniform(0,2,200)
        plt.hist(z, rwidth= 0.8)
        plt.show()

    if sys.argv[1] == "3.1":
        """ Softmax in TF """

        def softmax(x):
            e = tf.exp(x)
            row_wise_sum = tf.reduce_sum(e, axis = 1, keepdims= True)
            return e / row_wise_sum
        
        x = tf.constant([[-1, -1, 5], [1, 1, 3], [0, 0, 1]], dtype = tf.float32)
        print("Softmax:", softmax(x))

    if sys.argv[1] == "3.2":

        """ Computing gradients """
        
        def f(x):
            tmp = tf.range(1.0, x.shape[0] + 1)
            x = tf.math.multiply(x, x)
            return tf.reduce_sum(x * tmp)

        a1 = tf.constant([1., 2, 3])  
        a2 = tf.constant([2., 0, 2])  
        print(f(a1), f(a2))

        # B
        with tf.GradientTape(persistent=True) as g:
            g.watch(a1)
            y1 = f(a1)

        print(g.gradient(y1, a1))