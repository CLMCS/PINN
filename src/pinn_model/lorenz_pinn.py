import os
import time
import datetime
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from decimal import Decimal
from scipy import interpolate
from src import DenseBlock

class LorenzPINN(tf.keras.Model):
    def __init__(self, layers=4, layer_width=20, bn=False, log_opt=False, lr=1e-2):
       # initialize
        super(LorenzPINN, self).__init__()

        self.c1 = tf.Variable(1.0)
        self.c2 = tf.Variable(1.0)
        self.c3 = tf.Variable(1.0)

        self.NN = DenseBlock(layers, layer_width, bn)
        self.epochs = 0
        self.log_opt = log_opt
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.string = "{}_layers{}_neurons{}_bn{}_log{}".format(
            current_time, layers, layer_width, bn, log_opt
        )
        self.train_summary_writer = tf.summary.create_file_writer(
            "logs/" + self.string
        )

    @tf.function
    def call(self, t):
        # optimize in log-space
        vars = self.predict()

        with tf.GradientTape(persistent=True) as g:
            g.watch(t)
            [x, y, z] = self.NN(t)

        # gradients w.r.t. t
        dx_dt = g.gradient(x, t)
        dy_dt = g.gradient(y, t)
        dz_dt = g.gradient(z, t)
        del g


        fx = dx_dt - vars[0] * (y - x)
        fy = dy_dt - x * (vars[1] - z) + y
        fz = dz_dt - x * y + vars[2] * z

        return [x, y, z, fx, fy, fz]

    def set_lr(self, lr):
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    def get_loss(self, t, u_true, tuning_lambda):
        return self.__mse(u_true, self(t), tuning_lambda)

    def get_error(self, true):
        pred = tf.squeeze(self.predict())
        true = tf.convert_to_tensor(true)
        return tf.reduce_sum(tf.abs(pred - true))

    def predict(self):
        var_tensor_mu = tf.convert_to_tensor([self.c1,self.c2,self.c3])
        exp_var_tensor_mu = tf.exp(var_tensor_mu) if self.log_opt else var_tensor_mu

        return exp_var_tensor_mu

    def predict_curves(self,t):
        return self.NN(t)

    def optimize(self, t, u_true, tuning_lambda):
        # make the t here to be dense
        loss = lambda: self.get_loss(t, u_true, tuning_lambda)
        self.optimizer.minimize(loss=loss, var_list=self.trainable_weights)

    def save_model(self):

        self.save_weights("model_weights/{}.tf".format(self.string))

    def fit(self,observed_data,true_pars,epochs,tuning_lambda,verbose=False):

        for ep in range(self.epochs+1,self.epochs+epochs+1):
            # physical loss here is only evaluated at time t for t in the observation time points
            self.optimize(observed_data[0],[observed_data[1],observed_data[2],observed_data[3]], tuning_lambda)
            
            if ep % 100 == 0 or ep == 1:
                # pred = self.predict()
                loss = self.get_loss(observed_data[0], [observed_data[1],observed_data[2],observed_data[3]], tuning_lambda) / observed_data[0].shape[0]
                error = self.get_error(true_pars)        
                curves = self.predict_curves(observed_data[0])
                if verbose:
                    print('\n')
                    print(
                        "Epoch: {:5d}, loss: {:.2E}, error: {:3.2f}".format(
                            ep, Decimal(loss.numpy().item()), error.numpy().item()
                        )                    
                    )
                    print(
                        "c1: {:3.2f}, c2: {:3.2f}, c3: {:3.2f}".format(
                            np.exp(self.c1.numpy().item()), np.exp(self.c2.numpy().item()), np.exp(self.c3.numpy().item())
                        )                    
                    )

        self.epochs += epochs
        self.save_model()

    # def cv_lambda(self, data, data_noise, n):
    #     rng = np.random.RandomState(seed=5)
    #     all_indices = rng.permutation(n)
    #     train_indices = all_indices[:round(n * 0.8)]
    #     test_indices = all_indices[round(n * 0.2):]
    #     best_lambda = None
    #     best_error = None
    #     kfold = 10
    #     lambda_list = [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100, 1000, 10000]
    #     for lm in lambda_list:
    #         err = Regression().ridge_cross_validation(data[train_indices], data_noise[train_indices], kfold, lm)
    
    #         if best_error is None or err < best_error: 
    #             best_error = err
    #             best_lambda = lm
    #             weight = Regression().ridge_fit_closed(data[train_indices], data_noise[train_indices], c_lambda=10)
    #             y_test_pred = Regression().predict(data[test_indices], weight)
    #     return best_lambda

    def __mse(self, u_true, y_pred, tuning_lambda):

        # pred = self.predict()

        loss_x = tf.reduce_mean(tf.square(y_pred[0] - u_true[0]))
        loss_y = tf.reduce_mean(tf.square(y_pred[1] - u_true[1]))
        loss_z = tf.reduce_mean(tf.square(y_pred[2] - u_true[2]))
        loss_fx = tf.reduce_mean(tf.square(y_pred[3]))
        loss_fy = tf.reduce_mean(tf.square(y_pred[4]))
        loss_fz = tf.reduce_mean(tf.square(y_pred[5]))

        # loss_neg = tf.reduce_sum(tf.abs(tf.minimum(pred, 0.1) - 0.1))
        # 10 should be replaced by lambda tuning parameter, especailly for noisy observations
        return tuning_lambda*(loss_x + loss_y + loss_z) + loss_fx + loss_fy + loss_fz #+ loss_neg




