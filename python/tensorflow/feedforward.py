from typing import Callable, Tuple, Sequence

import tensorflow as tf
import time
import math


def input_expected_placeholders(batch_size: int) -> (tf.placeholder, tf.placeholder):
    inputs = tf.placeholder(tf.float32, shape=(batch_size, 1))
    expected = tf.placeholder(tf.float32, shape=(batch_size,))
    return inputs, expected


# TODO: Use a loop to build the network; then use same method to create
#       the quadratic feedforward network.
def linear_feedforward(input_layer, layers: Sequence[int]):
    prior_node_count = input_layer.get_shape()[1].value
    for i in range(len(layers)):
        last_layer = i == len(layers) - 1
        name = 'output_layer' if last_layer else 'layer_' + str(i)
        node_count = layers[i]
        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([prior_node_count, node_count],
                                    stddev=1.0 / math.sqrt(float(1))),
                name='fw')
            biases = tf.Variable(tf.zeros([node_count]),
                                 name='fb')
            input_layer = tf.matmul(input_layer, weights) + biases
            if not last_layer:
                input_layer = tf.nn.relu(input_layer)
            prior_node_count = node_count
    return input_layer


def quadratic_feedforward(input_layer, layers: Sequence[int], has_activation: bool = False):
    prior_node_count = input_layer.get_shape()[1].value
    for i in range(len(layers)):
        last_layer = i == len(layers) - 1
        name = 'output_layer' if last_layer else 'layer_' + str(i)
        node_count = layers[i]
        with tf.name_scope(name):
            fw = tf.Variable(
                tf.truncated_normal([prior_node_count, node_count],
                                    stddev=1.0 / math.sqrt(float(1))),
                name='fw')
            fb = tf.Variable(tf.zeros([node_count]),
                             name='fb')
            gw = tf.Variable(
                tf.truncated_normal([prior_node_count, node_count],
                                    stddev=1.0 / math.sqrt(float(1))),
                name='gw')
            gb = tf.Variable(tf.zeros([node_count]),
                             name='gb')
            input_layer = tf.mul(tf.matmul(input_layer, fw) + fb, tf.matmul(input_layer, gw) + gb)
            if has_activation and not last_layer:
                input_layer = tf.nn.relu(input_layer)
            prior_node_count = node_count

    return input_layer


def get_inputs(batch_size: int, input_range: Tuple[int, int]):
    min_val, max_val = input_range
    return tf.random_uniform((batch_size, 1), min_val, max_val)


def loss(actual_output, expected_output):
    return tf.nn.l2_loss(tf.sub(actual_output, expected_output))


def traditional_optimizer(loss, learning_rate):
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


def log_scaled_optimizer(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    gradients_and_variables = optimizer.compute_gradients(loss)
    log_gradients = [(tf.mul(tf.sign(grad), tf.log1p(tf.abs(grad))), var)
                     for grad, var in gradients_and_variables]
    return optimizer.apply_gradients(log_gradients, global_step=global_step)


if __name__ == "__main__":
    with tf.Graph().as_default():
        inputs = get_inputs(100, (-20, 20))
        expected = tf.mul(inputs, tf.sin(inputs))

        # Linear model
        # model = linear_feedforward(inputs, [2, 2, 1])
        # loss = loss(model, expected)
        # trainer = traditional_optimizer(loss, .001)

        # Quadratic model
        model = quadratic_feedforward(inputs, [2, 2, 2, 1])
        loss = loss(model, expected)
        trainer = log_scaled_optimizer(loss, .001)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter('D:\\git\\domenic\\python\\tensorflow\\logs',
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
            _, loss_value = sess.run([trainer, tf.log1p(loss)])

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary)
                print(summary_str)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
