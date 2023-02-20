import tensorflow as tf

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