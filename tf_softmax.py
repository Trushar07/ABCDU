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