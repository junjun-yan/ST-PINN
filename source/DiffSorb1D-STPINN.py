import time
import numpy as np
import tensorflow as tf
from source.pdes import Diffusion_sorption, Boundary
from source.utilities import tf_session, mean_squared_error, relative_error

np.random.seed(1234)
tf.set_random_seed(1234)


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

        return tf.nn.relu(Y)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data,
                 t_data, u_data, x_test, t_test, u_test, batch_size, layers, log_path, update_freq, max_rate,
                 stab_coeff):

        # points
        self.x_init, self.t_init, self.u_init = x_init, t_init, u_init
        self.x_l_bound, self.x_r_bound, self.t_bound = x_l_bound, x_r_bound, t_bound
        self.x_sample, self.t_sample = x_sample, t_sample
        self.x_data, self.t_data, self.u_data = x_data, t_data, u_data
        self.x_test, self.t_test, self.u_test = x_test, t_test, u_test

        # for pseudo flag
        self.flag_pseudo = np.zeros(shape=self.x_sample.shape, dtype=np.int32)

        self.layers = layers
        self.log_path = log_path
        self.batch_size = batch_size
        self.update_freq, self.max_rate, self.stab_coeff = update_freq, max_rate, stab_coeff

        # tf Placeholders
        self.x_init_tf, self.t_init_tf, self.u_init_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        self.x_l_bound_tf, self.x_r_bound_tf, self.t_bound_tf = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        self.x_data_tf, self.t_data_tf, self.u_data_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        self.x_eqns_tf, self.t_eqns_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]
        self.x_pseudo_tf, self.t_pseudo_tf, self.u_pseudo_tf = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]

        # tf Graphs
        self.net = neural_net(self.x_sample, self.t_sample, layers=self.layers)

        self.u_init_pred = self.net(self.x_init_tf, self.t_init_tf)
        self.u_l_bound_pred = self.net(self.x_l_bound_tf, self.t_bound_tf)
        self.u_r_bound_pred = self.net(self.x_r_bound_tf, self.t_bound_tf)
        self.u_data_pred = self.net(self.x_data_tf, self.t_data_tf)
        self.u_eqns_pred = self.net(self.x_eqns_tf, self.t_eqns_tf)
        self.u_pseudo_pred = self.net(self.x_pseudo_tf, self.t_pseudo_tf)
        self.e = Diffusion_sorption(self.x_eqns_tf, self.t_eqns_tf, self.u_eqns_pred)
        self.b = Boundary(self.x_r_bound_tf, self.u_r_bound_pred)

        # Loss
        self.init_loss = mean_squared_error(self.u_init_pred, self.u_init_tf)
        self.bound_loss = mean_squared_error(self.u_l_bound_pred, 1.0) + mean_squared_error(self.u_r_bound_pred, self.b)
        self.eqns_loss = mean_squared_error(self.e, 0)
        self.data_loss = mean_squared_error(self.u_data_pred, self.u_data_tf)
        self.pseudo_loss = mean_squared_error(self.u_pseudo_pred, self.u_pseudo_tf)

        self.loss = self.init_loss + self.eqns_loss + 10 * self.data_loss + self.bound_loss + self.pseudo_loss

        # Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf_session()

    def train(self, max_time, adam_it):

        self.it = 0
        self.total_time = 0
        self.start_time = time.time()
        self.update_data_eqns_points()

        while self.it < adam_it and self.total_time < max_time:

            N_eqns = self.t_eqns.shape[0]
            idx_batch = np.random.choice(N_eqns, min(self.batch_size, N_eqns), replace=False)
            x_eqns_batch = self.x_eqns[idx_batch, :]
            t_eqns_batch = self.t_eqns[idx_batch, :]

            N_pseudo = self.t_pseudo.shape[0]
            idx_batch = np.random.choice(N_pseudo, min(self.batch_size, N_pseudo), replace=False)
            x_pseudo_batch = self.x_pseudo[idx_batch, :]
            t_pseudo_batch = self.t_pseudo[idx_batch, :]
            u_pseudo_batch = self.u_pseudo[idx_batch, :]

            tf_dict = {self.x_init_tf: self.x_init, self.t_init_tf: self.t_init, self.u_init_tf: self.u_init,
                       self.x_l_bound_tf: self.x_l_bound, self.x_r_bound_tf: self.x_r_bound,
                       self.t_bound_tf: self.t_bound,
                       self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.u_data_tf: self.u_data,
                       self.x_eqns_tf: x_eqns_batch, self.t_eqns_tf: t_eqns_batch,
                       self.x_pseudo_tf: x_pseudo_batch, self.t_pseudo_tf: t_pseudo_batch,
                       self.u_pseudo_tf: u_pseudo_batch}

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0
                loss_value, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss = self.sess.run(
                    [self.loss, self.init_loss, self.bound_loss, self.eqns_loss, self.data_loss, self.pseudo_loss],
                    tf_dict)
                log_item = 'It: %d, Loss: %.3e, Init Loss: %.3e, Bound Loss: %.3e, Eqns Loss: %.3e, ' \
                           'Data Loss: %.3e, Pseudo Loss: %.3e, Time: %.2fs, Total Time: %.2fh' % \
                           (self.it, loss_value, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss, elapsed,
                            self.total_time)
                self.logging(log_item)
                self.start_time = time.time()

            # evaluate
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test)
                log_item = 'Error u: %e' % (error_u)
                self.logging(log_item)

            # update pseudo points
            if self.it % self.update_freq == 0:
                self.update_data_eqns_points()

            self.it += 1

    def update_data_eqns_points(self):
        # compute equation residual error for all sample points
        tf_dict = {self.x_eqns_tf: self.x_sample, self.t_eqns_tf: self.t_sample}
        [e] = np.abs(self.sess.run(self.e, tf_dict))

        # get eqns index and pseudo index, sorted by ns loss
        sample_size = self.t_sample.shape[0]
        pseudo_size = int(self.max_rate * sample_size)

        if pseudo_size == 0:
            idx_pseudo = []
        else:
            idx_pseudo = np.argpartition(e.T[0], pseudo_size)[:pseudo_size]

        self.flag_pseudo[idx_pseudo] += 1
        mask = np.ones(self.flag_pseudo.shape[0], np.bool8)
        mask[idx_pseudo] = False
        self.flag_pseudo[mask] = 0

        idx_pseudo = np.where(self.flag_pseudo > self.stab_coeff)[0]
        self.t_pseudo = self.t_sample[idx_pseudo, :]
        self.x_pseudo = self.x_sample[idx_pseudo, :]

        # generate pseudo points
        tf_dict = {self.x_pseudo_tf: self.x_pseudo, self.t_pseudo_tf: self.t_pseudo}
        [self.u_pseudo] = self.sess.run(self.u_pseudo_pred, tf_dict)

        mask = np.ones(self.flag_pseudo.shape[0], np.bool8)
        mask[idx_pseudo] = False
        self.t_eqns = self.t_sample[mask]
        self.x_eqns = self.x_sample[mask]

        pseudo_size = self.t_pseudo.shape[0]
        eqns_size = self.t_eqns.shape[0]
        self.logging("Number Pseudo Points : %d, Number Eqns Points : %d" % (pseudo_size, eqns_size))

    def predict(self, x_star, t_star):
        tf_dict = {self.x_eqns_tf: x_star, self.t_eqns_tf: t_star}
        u_star = self.sess.run(self.u_eqns_pred, tf_dict)[0]
        return u_star

    def logging(self, log_item):
        with open(self.log_path, 'a+') as log:
            log.write(log_item + '\n')
        print(log_item)


if __name__ == '__main__':
    xL, xR = 0, 1
    N_init = 5000
    N_bound = 1000
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "../output/log/diffsorb1D-stpinn-%s" % (create_date)

    # pseudo
    update_freq = 100
    max_rate = 0.999
    stab_coeff = 1

    ### load data
    data_path = r'../input/diffsorb1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    # init
    idx_init = np.where(t == 0.0)[0]
    x_init = x[idx_init, :]
    t_init = t[idx_init, :]
    u_init = u[idx_init, :]

    # boundary
    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound, :]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    ### rearrange data
    # eqns
    x_sample = x
    t_sample = t

    # initail
    idx_init = np.random.choice(x_init.shape[0], min(N_init, x_init.shape[0]), replace=False)
    x_init = x_init[idx_init, :]
    t_init = t_init[idx_init, :]
    u_init = u_init[idx_init, :]

    # boundary
    idx_bound = np.random.choice(t_bound.shape[0], min(N_bound, t_bound.shape[0]), replace=False)
    x_l_bound = x_l_bound[idx_bound, :]
    x_r_bound = x_r_bound[idx_bound, :]
    t_bound = t_bound[idx_bound, :]

    # intre-domain
    idx_data = np.random.choice(x.shape[0], min(N_data, x.shape[0]), replace=False)
    x_data = x[idx_data, :]
    t_data = t[idx_data, :]
    u_data = u[idx_data, :]

    # test
    idx_test = np.random.choice(x.shape[0], min(N_test, x.shape[0]), replace=False)
    x_test = x[idx_test, :]
    t_test = t[idx_test, :]
    u_test = u[idx_test, :]

    ## build model
    model = PhysicsInformedNN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data, t_data,
                              u_data, x_test, t_test, u_test, batch_size, layers, log_path, update_freq, max_rate,
                              stab_coeff)

    ### train
    model.train(max_time=10, adam_it=20000)

    ### test
    u_pred = model.predict(x, t)
    error_u = relative_error(u_pred, u)
    model.logging('L2 error u: %e' % (error_u))

    u_pred = model.predict(x, t)
    error_u = mean_squared_error(u_pred, u)
    model.logging('MSE error u: %e' % (error_u))

    # save prediction
    data_output_path = "../output/prediction/diffsorb1D-stpinn-%s.npy" % (create_date)
    data_output = {'u': u_pred}
    np.save(data_output_path, data_output)
