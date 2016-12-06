import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.zeros([1, 10]))
b1 = tf.Variable(tf.zeros([10]))

W2 = tf.Variable(tf.zeros([10, 10]))
b2 = tf.Variable(tf.zeros([10]))

W3 = tf.Variable(tf.zeros([10, 1]))
b3 = tf.Variable(tf.zeros([1]))

x1 = tf.nn.relu(tf.matmul(x, W1) + b1)
x2 = tf.nn.relu(tf.matmul(x1, W2) + b2)
y_ = tf.nn.relu(tf.matmul(x2, W3) + b3)