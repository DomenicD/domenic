from typing import Callable, Tuple

import tensorflow as tf
import time
import math


def input_expected_placeholders(batch_size: int) -> (tf.placeholder, tf.placeholder):
    inputs = tf.placeholder(tf.float32, shape=(batch_size, 1))
    expected = tf.placeholder(tf.float32, shape=(batch_size,))
    return inputs, expected


def linear_feedforward(inputs, hidden1_units: int, hidden2_units: int):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([1, hidden1_units],
                                stddev=1.0 / math.sqrt(float(1))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('linear_output'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, 1],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([1]),
                             name='biases')
        outputs = tf.matmul(hidden2, weights) + biases
    return outputs


def quadratic_feedforward():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    # Forces to run on CPU.
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            print(sess.run(product))


def get_inputs(batch_size: int, range: Tuple[int, int]):
    min, max = range
    return tf.random_uniform((batch_size, 1), min, max)


def loss(actual, expected):
    return tf.nn.l2_loss(tf.sub(actual, expected))


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


if __name__ == "__main__":
    with tf.Graph().as_default():
        inputs = get_inputs(10, (-5, 5))
        expected = tf.mul(inputs, tf.sin(inputs))
        model = linear_feedforward(inputs, 5, 5)
        loss = loss(model, expected)
        trainer = training(loss, .001)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter('D:\\git\\domenic\\python\\tensorflow\\logs',
                                                sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(2000):
            start_time = time.time()

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([trainer, loss])

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
