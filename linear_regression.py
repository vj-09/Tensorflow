import tensorflow as tf
import numpy as np


a = tf.Variable(2.0)
b = tf.Variable(2.0)

x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

y= a * x + b
loss = tf.reduce_mean(tf.square(y - d))
optimizer = tf.train.GradientDescentOptimizer(0.005)

train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(100000):
    sess.run([train], feed_dict={ x:[1.0,2.0,3.0], d:[6.5,12.5,18.5]} )
    if step % 5 == 0:
        print( step, sess.run([a,b]) ) #print step and value of a & b

sess.close()
