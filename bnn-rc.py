import traceback

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from tensorflow.python.framework import ops

from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer, concatenate, Lambda, Dense

import tensorflow as tf
from pandas import read_csv

from keras.layers import Input
from keras.models import Model, load_model

from keras import optimizers
from keras.callbacks import EarlyStopping, TerminateOnNaN

import tqdm


def main(training_file, testing_file_list):
    print(f"training_file={training_file}")
    print(f"testing_file_list={testing_file_list}")
    print("------")
    K.clear_session()

    def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
        params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
        sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
        return params, sigma

    prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=.1, pi=0.2)

    print(f"prior_params={prior_params}")
    print(f"prior_sigma={prior_sigma}")

    def log_mixture_prior_prob(w):
        comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
        comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
        comp_1_weight = prior_params[2]
        return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))

    class DenseVariational(Layer):
        def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
            self.output_dim = output_dim
            self.kl_loss_weight = kl_loss_weight
            self.activation = activations.get(activation)
            self.kernel_mu = self.bias_mu = self.kernel_rho = self.bias_rho = None
            super().__init__(**kwargs)

        def build(self, input_shape):
            self._trainable_weights.append(prior_params)
            self.kernel_mu = self.add_weight(name='kernel_mu',
                                             shape=(input_shape[1], self.output_dim),
                                             initializer=initializers.normal(stddev=prior_sigma),
                                             trainable=True)
            self.bias_mu = self.add_weight(name='bias_mu',
                                           shape=(self.output_dim,),
                                           initializer=initializers.normal(stddev=prior_sigma),
                                           trainable=True)
            self.kernel_rho = self.add_weight(name='kernel_rho',
                                              shape=(input_shape[1], self.output_dim),
                                              initializer=initializers.constant(0.0),
                                              trainable=True)
            self.bias_rho = self.add_weight(name='bias_rho',
                                            shape=(self.output_dim,),
                                            initializer=initializers.constant(0.0),
                                            trainable=True)
            super().build(input_shape)

        def call(self, x):
            kernel_sigma = tf.math.softplus(self.kernel_rho)
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

            bias_sigma = tf.math.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

            self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                          self.kl_loss(bias, self.bias_mu, bias_sigma))

            return self.activation(K.dot(x, kernel) + bias)

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.output_dim

        def kl_loss(self, w, mu, sigma):
            variational_dist = tf.distributions.Normal(mu, sigma)
            return self.kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))

    # input includes current value T_set and previous T_in
    def active_q(inputs):
        current_t_set = inputs[0][:, 1]
        previous_t_in = inputs[1][:, 2]

        return K.reshape(K.cast(K.greater(current_t_set, previous_t_in), float) - \
                         K.cast(K.less(current_t_set, previous_t_in), float), (-1, 1))

    def rc_keras(inputs):
        historical = inputs[1]
        inputs = K.mean(inputs[0], axis=0) + 1e-6

        timestep = 1
        cap_a = K.zeros((3, 3), dtype=float)
        cap_b = K.zeros((3, 2), dtype=float)
        cap_c = K.zeros((1, 3), dtype=float)
        cap_d = K.zeros((1, 2), dtype=float)

        cap_a = K.update_add(cap_a, [[-1 / (inputs[4] * inputs[0]) - 1 / (inputs[4] * inputs[1]),
                                      1 / (inputs[4] * inputs[1]),
                                      0],
                                     [1 / (inputs[5] * inputs[1]),
                                      -1 / (inputs[5] * inputs[1]) - 1 / (inputs[5] * inputs[2]),
                                      1 / (inputs[5] * inputs[2])],
                                     [0,
                                      1 / (inputs[6] * inputs[2]),
                                      -1 / (inputs[6] * inputs[2])]])
        cap_b = K.update_add(cap_b, [[1 / (inputs[4] * inputs[0]), 0],
                                     [0, 0],
                                     [0, inputs[3] / inputs[6]]])
        cap_c = K.update_add(cap_c, [[0, 0, 1]])

        phi = tf.linalg.expm(cap_a * timestep)
        # phi = cap_a
        cap_i = K.eye(3)

        gamma1 = K.dot(K.dot(tf.linalg.inv(cap_a), phi - cap_i), cap_b)
        gamma2 = K.dot(tf.linalg.inv(cap_a), gamma1 / timestep - cap_b)
        # gamma1 = K.dot(K.dot(cap_a, phi - cap_i), cap_b)
        # gamma2 = K.dot(cap_a, gamma1 / timestep - cap_b)

        cap_r0 = cap_i
        mul_result = K.dot(phi, cap_r0)
        e1 = -tf.linalg.trace(mul_result) / 1

        cap_r1 = mul_result + e1 * cap_i
        mul_result = K.dot(phi, cap_r1)
        e2 = -tf.linalg.trace(mul_result) / 2

        cap_r2 = mul_result + e2 * cap_i
        mul_result = K.dot(phi, cap_r2)
        e3 = -tf.linalg.trace(mul_result) / 3

        cap_s0 = K.dot(K.dot(cap_c, cap_r0), gamma2) + cap_d
        cap_s1 = K.dot(cap_c, K.dot(cap_r0, gamma1 - gamma2) + K.dot(cap_r1, gamma2)) + e1 * cap_d
        cap_s2 = K.dot(cap_c, K.dot(cap_r1, gamma1 - gamma2) + K.dot(cap_r2, gamma2)) + e2 * cap_d
        cap_s3 = K.dot(cap_c, K.dot(cap_r2, gamma1 - gamma2)) + e3 * cap_d

        y_vector = K.concatenate((cap_s0, [[-e1]], cap_s1, [[-e2]], cap_s2, [[-e3]], cap_s3), axis=1)

        y_vector = y_vector / K.sum(y_vector)
        result = K.dot(historical, K.transpose(y_vector))

        # y_in_t = K.dot(cap_s0, K.variable([[historical[0]], [historical[1]]])) + \
        #          K.dot(cap_s1, K.variable([[historical[3]], [historical[4]]])) + \
        #          K.dot(cap_s2, K.variable([[historical[6]], [historical[7]]])) + \
        #          K.dot(cap_s3, K.variable([[historical[9]], [historical[10]]])) - \
        #          e1 * historical[2] - e2 * historical[5] - e3 * historical[8]
        return result

    def rc_cap_a(inputs):
        inputs = K.mean(inputs[0], axis=0)
        inputs += tf.random.normal(inputs.shape)

    def loss_wrapper(layer):
        def loss_rc(y_true, rc_pred):
            historical = layer
            inputs = K.mean(rc_pred, axis=0) + 1e-10

            timestep = 1
            cap_a = K.zeros((3, 3), dtype=float)
            cap_b = K.zeros((3, 2), dtype=float)
            cap_c = K.zeros((1, 3), dtype=float)
            cap_d = K.zeros((1, 2), dtype=float)

            cap_a = K.update_add(cap_a, [[-1 / (inputs[4] * inputs[0]) - 1 / (inputs[4] * inputs[1]),
                                          1 / (inputs[4] * inputs[1]),
                                          0],
                                         [1 / (inputs[5] * inputs[1]),
                                          -1 / (inputs[5] * inputs[1]) - 1 / (inputs[5] * inputs[2]),
                                          1 / (inputs[5] * inputs[2])],
                                         [0,
                                          1 / (inputs[6] * inputs[2]),
                                          -1 / (inputs[6] * inputs[2])]])
            cap_b = K.update_add(cap_b, [[1 / (inputs[4] * inputs[0]), 0],
                                         [0, 0],
                                         [0, inputs[3] / inputs[6]]])
            cap_c = K.update_add(cap_c, [[0, 0, 1]])

            phi = tf.linalg.expm(cap_a * timestep)
            # phi = cap_a
            cap_i = K.eye(3)

            gamma1 = K.dot(K.dot(tf.linalg.inv(cap_a), phi - cap_i), cap_b)
            gamma2 = K.dot(tf.linalg.inv(cap_a), gamma1 / timestep - cap_b)
            # gamma1 = K.dot(K.dot(cap_a, phi - cap_i), cap_b)
            # gamma2 = K.dot(cap_a, gamma1 / timestep - cap_b)

            cap_r0 = cap_i
            mul_result = K.dot(phi, cap_r0)
            e1 = -tf.linalg.trace(mul_result) / 1

            cap_r1 = mul_result + e1 * cap_i
            mul_result = K.dot(phi, cap_r1)
            e2 = -tf.linalg.trace(mul_result) / 2

            cap_r2 = mul_result + e2 * cap_i
            mul_result = K.dot(phi, cap_r2)
            e3 = -tf.linalg.trace(mul_result) / 3

            cap_s0 = K.dot(K.dot(cap_c, cap_r0), gamma2) + cap_d
            cap_s1 = K.dot(cap_c, K.dot(cap_r0, gamma1 - gamma2) + K.dot(cap_r1, gamma2)) + e1 * cap_d
            cap_s2 = K.dot(cap_c, K.dot(cap_r1, gamma1 - gamma2) + K.dot(cap_r2, gamma2)) + e2 * cap_d
            cap_s3 = K.dot(cap_c, K.dot(cap_r2, gamma1 - gamma2)) + e3 * cap_d

            y_vector = K.concatenate((cap_s0, [[-e1]], cap_s1, [[-e2]], cap_s2, [[-e3]], cap_s3), axis=1)

            # y_vector = y_vector / K.sum(y_vector)
            result = K.dot(historical, K.transpose(y_vector))

            # y_in_t = K.dot(cap_s0, K.variable([[historical[0]], [historical[1]]])) + \
            #          K.dot(cap_s1, K.variable([[historical[3]], [historical[4]]])) + \
            #          K.dot(cap_s2, K.variable([[historical[6]], [historical[7]]])) + \
            #          K.dot(cap_s3, K.variable([[historical[9]], [historical[10]]])) - \
            #          e1 * historical[2] - e2 * historical[5] - e3 * historical[8]
            return y_true - result

        return loss_rc

    noise = 1.0
    vb = 2
    epo = 50
    run_pred = 5

    def neg_log_likelihood(y_true, y_pred, sigma=noise):
        dist = tf.distributions.Normal(loc=y_pred, scale=sigma)
        return K.sum(-dist.log_prob(y_true))

    split_point = 30000
    retrain_point = 300
    batch_size = 12
    model_callbacks = [EarlyStopping(monitor='loss', patience=2,
                                     restore_best_weights=True,
                                     min_delta=0.02),
                       TerminateOnNaN()]

    data = read_csv(training_file,
                    header=0,
                    parse_dates=[0],
                    index_col=0,
                    ).values
    data = np.concatenate((data[3:, :], data[2:-1, :], data[1:-2, :], data[:-3, :]), axis=1)

    X_1 = data[:, 1:]
    X_2 = data[:, 1:]
    y = data[:, 0].reshape((-1, 1))

    train_size = X_1.shape[0]
    num_batches = train_size / batch_size
    kl_loss_weight = 1.0 / num_batches

    # Switch the model structure depends on the needs
    x_in = Input(shape=(X_1.shape[1],))
    # x = Dense(7, activation='relu')(x_in)
    x_hist = Input(shape=(X_2.shape[1],))
    x = DenseVariational(1, kl_loss_weight=kl_loss_weight, activation='relu')(x_in)
    # x = DenseVariational(7, kl_loss_weight=kl_loss_weight, activation='relu')(x)
    # x = DenseVariational(7, kl_loss_weight=kl_loss_weight)(x_in)
    # k = Lambda(active_q, output_shape=(1,))([x, x_hist])
    # x = Lambda(rc_keras, output_shape=(1,))([x, x_hist])
    # x = concatenate([x, x_in])
    # x = Dense(1)(x)
    model = Model(inputs=[x_in, x_hist], outputs=x)

    # model.summary()

    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])

    if not os.path.isfile("models/" + training_file[:-4] + "-weights.npy"):
        # model.compile(loss=my_loss_wrapper(x_hist), optimizer=optimizers.Adam(lr=0.001), metrics=['mse'])
        print("Training on %s with 300 data" % training_file)
        train_result = model.fit({"input_1": X_1[:retrain_point],
                                  "input_2": X_2[:retrain_point]},
                                 y[:retrain_point],
                                 batch_size=batch_size, epochs=epo, verbose=vb, callbacks=model_callbacks)

        if np.isnan(train_result.history['loss'][-1]) or np.isinf(train_result.history['loss'][-1]):
            print("Failed")
            return False

        y_pred_limit_list = []
        print("Testing on %s with 300 data" % training_file)
        for i in range(run_pred):
            y_pred_limit = model.predict([X_1, X_2])
            y_pred_limit_list.append(y_pred_limit)
        y_preds_limit = np.concatenate(y_pred_limit_list, axis=1)
        y_mean_limit = np.mean(y_preds_limit, axis=1)

        rmse_limit = (((y_mean_limit - y.flatten()) ** 2).mean()) ** 0.5

        print("Training on %s with 30000 data" % training_file)
        train_result = model.fit({"input_1": X_1[retrain_point:split_point],
                                  "input_2": X_2[retrain_point:split_point]},
                                 y[retrain_point:split_point],
                                 batch_size=batch_size, epochs=epo, verbose=vb, callbacks=model_callbacks)

        if np.isnan(train_result.history['loss'][-1]) or np.isinf(train_result.history['loss'][-1]):
            print("Failed")
            return False

        y_pred_list = []
        print("Testing on %s with 30000 data" % training_file)
        for i in range(run_pred):
            y_pred = model.predict([X_1, X_2])
            y_pred_list.append(y_pred)
        y_preds = np.concatenate(y_pred_list, axis=1)
        y_mean = np.mean(y_preds, axis=1)

        rmse = (((y_mean - y.flatten()) ** 2).mean()) ** 0.5
        print("---------- Testing on %s done ----------" % training_file)
        weights_bias = model.get_weights()

        with open("self.csv", 'a') as outfile:
            outfile.write(training_file + ',' + str(rmse_limit) + ',' + str(rmse))
            for weights in weights_bias:
                outfile.write(',' + ','.join(list(map(str, list(weights.flatten())))))
            outfile.write('\n')

        print("Saving model")
        if not os.path.isdir("models/" + training_file[:max(0, training_file.rfind('/'))]):
            os.makedirs("models/" + training_file[:max(0, training_file.rfind('/'))], 755)

        model.save_weights("models/" + training_file[:-4] + "-weights.npy")

        print("Weight saved")
    else:
        print("Load model")
        model.load_weights("models/" + training_file[:-4] + "-weights.npy")
        print("Model Loaded")

    print("Done self training")

    for testing_file in testing_file_list:
        finished_file = []
        if os.path.isfile("helper/" + training_file):
            with open("helper/" + training_file, 'r') as read:
                for line in read:
                    finished_file.extend(line.split(','))
        try:
            if testing_file == training_file or testing_file in finished_file:
                continue
            model.load_weights("models/" + training_file[:-4] + "-weights.npy")
            data = read_csv(testing_file,
                            header=0,
                            parse_dates=[0],
                            index_col=0,
                            ).values
            data[1:, 2][data[1:, 2] < data[:-1, 0]] = -1
            data[1:, 2][data[1:, 2] == data[:-1, 0]] = 0
            data[1:, 2][data[1:, 2] > data[:-1, 0]] = 1
            data[0, 2] = 0
            data = np.concatenate((data[3:, :], data[2:-1, :], data[1:-2, :], data[:-3, :]), axis=1)

            X_test_1 = data[:, 1:]
            X_test_2 = data[:, 1:]
            y_true = data[:, 0].reshape((-1, 1))

            y_pred_list = []
            print("Testing %s - %s" % (training_file, testing_file))
            for i in range(run_pred):
                y_pred = model.predict([X_test_1, X_test_2])
                y_pred_list.append(y_pred)

            y_preds = np.concatenate(y_pred_list, axis=1)
            y_mean = np.mean(y_preds, axis=1)

            rmse_empty = (((y_mean - y_true.flatten()) ** 2).mean()) ** 0.5

            print("Retraining %s - %s" % (training_file, testing_file))
            train_result = model.fit({"input_1": X_test_1[:retrain_point],
                                      "input_2": X_test_2[:retrain_point]},
                                     y_true[:retrain_point],
                                     batch_size=batch_size, epochs=epo, verbose=vb, callbacks=model_callbacks)

            if np.isnan(train_result.history['loss'][-1]) or np.isinf(train_result.history['loss'][-1]):
                continue

            print("Testing %s - %s with retraining" % (training_file, testing_file))
            for i in range(run_pred):
                y_pred = model.predict([X_test_1, X_test_2])
                y_pred_list.append(y_pred)
            y_preds = np.concatenate(y_pred_list, axis=1)
            y_mean = np.mean(y_preds, axis=1)

            rmse = (((y_mean - y_true.flatten()) ** 2).mean()) ** 0.5
            weights_bias = model.get_weights()

            # print("Saving model")
            if not os.path.isdir("cross-result/" + training_file[:max(0, training_file.rfind('/'))]):
                os.makedirs("cross-result/" + training_file[:max(0, training_file.rfind('/'))], 755)

            with open("cross-result/" + training_file, 'a') as outresult:
                outresult.write("%s,%f,%f" % (testing_file, rmse_empty, rmse))
                for weights in weights_bias:
                    outresult.write(',' + ','.join(list(map(str, list(weights.flatten())))))
                outresult.write('\n')

            if not os.path.isdir("helper/" + training_file[:max(0, training_file.rfind('/'))]):
                os.makedirs("helper/" + training_file[:max(0, training_file.rfind('/'))], 755)

            with open("helper/" + training_file, 'a') as outresult:
                outresult.write("%s," % testing_file)
        except:
            continue

    return True


def find_cluster_and_rest(not_belongs, threshold):
    not_belongs = not_belongs[:]  # Make a copy so won't hurt original list
    cluster_count = 0
    clusters = dict()

    while len(not_belongs) > 0:
        cluster_count += 1

        train_id = 0
        while train_id < len(not_belongs):
            data = read_csv(not_belongs[train_id],
                            header=0,
                            parse_dates=[0],
                            index_col=0,
                            ).values
            if data.shape[0] < 30000:
                train_id += 1
            else:
                break

        if train_id == len(not_belongs):
            break

        train = not_belongs.pop(train_id)

        main(train, not_belongs)

        data = read_csv("cross-result/" + train).values
        results = list(data[:, 0][data[:, 2] >= threshold])
        not_belongs = list(set(not_belongs) - set(results))
        not_belongs.sort()
        clusters[train] = results[:]

    print("Clusters:")
    print(clusters)
    print("Self cluster:")
    print(not_belongs)

    return True


if __name__ == '__main__':
    alist = ["cluster/" + 'clustercenter1.csv', "cluster/" + 'clustercenter1member.csv',
             "cluster/" + 'clustercenter2.csv', "cluster/" + 'clustercenter2member.csv',
             "cluster/" + 'clusterc27-1.csv', "cluster/" + 'clusterc27-4.csv', "cluster/" + 'clusterc27-5.csv',
             "cluster/" + 'clusterc27-2.csv', "cluster/" + 'clusterc27-3.csv']
    #
    num_of_process = 10
    process_id = 0  # 0 ~ 9

    length = len(alist)
    process_jobs = length // 5

    for i in range(process_jobs):
        try:
            main(alist[process_id * process_jobs + i], alist[process_id * process_jobs:(process_id + 1) * process_jobs])
        except:
            traceback.print_exc()
            print("Unknown Error!")
