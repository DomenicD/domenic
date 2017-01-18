import tensorflow as tf


def gpu_session():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    # Will run on GPU if available.
    with tf.Session() as sess:
        print(sess.run(product))


def cpu_session():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    # Forces to run on CPU.
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            print(sess.run(product))


def debug():
    sess = tf.InteractiveSession()
    a = tf.convert_to_tensor([[1, 2, 3], [2, 2, 2]])
    print(tf.rank(a).eval())
    sess.close()


if __name__ == "__main__":
    debug()
