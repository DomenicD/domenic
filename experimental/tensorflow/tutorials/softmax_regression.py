import tensorflow as tf
import gzip, pickle, numpy

with gzip.open("/home/djdonato/Downloads/mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

def expand_label(y):
    return [1 if n == y else 0 for n in range(10)]

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# TODO(domenic): visualize graph to understand how this modifies the computation graph.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

train_x, train_y = train_set
# Convert y from number to an array of size 10 with all zeros and 1 one.
# eg. 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
train_y = [expand_label(y) for y in train_y]
# Convert to numpy arrays so we can do multi index selection.
# eg. train_x[[1,3,5]] -> will create new array with elements
# from indices 1, 3, and 5.
train_x = numpy.array(train_x)
train_y = numpy.array(train_y)
train_size = len(train_x)

for i in range(1000):
    # Shuffle the data
    sample_indices = numpy.arange(train_size)
    numpy.random.shuffle(sample_indices)
    sample_indices = sample_indices[:100]
    sess.run(train_step, feed_dict={x: train_x[sample_indices], y_: train_y[sample_indices]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

valid_x, valid_y = valid_set
valid_y = [expand_label(y) for y in valid_y]

print(sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}))