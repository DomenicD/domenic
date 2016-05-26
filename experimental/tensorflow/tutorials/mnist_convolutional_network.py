import tensorflow as tf
import gzip, pickle, numpy

with gzip.open("/home/djdonato/Downloads/mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


# Prepare data
# TODO(domenic): This could be part of the computational graph.
# bonus: create a tensorflow graph to perform this data transform.
def to_one_hot(y_list):
    return numpy.array([[1 if n == y else 0 for n in range(10)] for y in y_list])

# Convert y from number to an array of size 10 with all zeros and 1 one.
# eg. 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
train_x, train_y = train_set
train_x = numpy.array(train_x)
train_y = numpy.array(train_y)
train_y = to_one_hot(train_y)

valid_x, valid_y = valid_set
valid_x = numpy.array(valid_x)
valid_y = numpy.array(valid_y)
valid_y = to_one_hot(valid_y)

test_x, test_y = test_set
test_x = numpy.array(test_x)
test_y = numpy.array(test_y)
test_y = to_one_hot(test_y)


# Create computational graph
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# TODO(domenic): What does -1 mean in Shape.
# https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer (same as softmax regression)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train the network
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# TODO(domenic): visualize graph to understand how this modifies the computation graph.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train_size = len(train_x)

for i in range(20000):
    # Shuffle the data
    sample_indices = numpy.arange(train_size)
    numpy.random.shuffle(sample_indices)
    sample_indices = sample_indices[:50]

    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: train_x[sample_indices], y_: train_y[sample_indices], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    sess.run(train_step,
             feed_dict={x: train_x[sample_indices], y_: train_y[sample_indices], keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y, keep_prob: 1.0}))
