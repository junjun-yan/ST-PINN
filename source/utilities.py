import numpy as np
import tensorflow as tf


class neural_net(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def __call__(self, *inputs):

        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if l < self.num_layers - 2:
                H = tf.nn.tanh(H)

        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)

        return Y


def mean_squared_error(pred, exact):
    # mse
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(tf.subtract(pred, exact)))


def relative_error(pred, exact):
    # l2 error
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact)))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact)))


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess
