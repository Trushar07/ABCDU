import tensorflow as tf

def MSE(Y, T):
    return tf.reduce_mean(tf.square(Y - T))

Y = tf.constant([[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.8, 0.1, 0.1]])
T = tf.constant([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype= float)
print(MSE(Y, T))
print(MSE(Y[0:2], T[0:2]))