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

    def get_loss(self, t_obs, u_true, t_dense, tuning_lambda):
        mse_part_xyz = self.__mse_xyz(u_true, self(t_obs))
        mse_part_dxdydz = self.__mse_dzdydz(self(t_dense))
        # mse_part_xyz = np.repeat(mse_part_xyz, len(mse_part_dxdydz)/len(mse_part_xyz), axis=0)
        return tuning_lambda * mse_part_xyz + mse_part_dxdydz

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

    def optimize(self, t, t_dense, u_true, tuning_lambda):
        # make the t here to be dense
        loss = lambda: self.get_loss(t, u_true, t_dense, tuning_lambda)
        self.optimizer.minimize(loss=loss, var_list=self.trainable_weights)

    def save_model(self):

        self.save_weights("model_weights/{}.tf".format(self.string))

    def fit(self,observed_data,true_pars,epochs,dense_time,tuning_lambda,verbose=False):

        for ep in range(self.epochs+1,self.epochs+epochs+1):
            # physical loss here is only evaluated at time t for t in the observation time points
            self.optimize(observed_data[0],dense_time,[observed_data[1],observed_data[2],observed_data[3]], tuning_lambda)
            
            if ep % 100 == 0 or ep == 1:
                # pred = self.predict()
                loss = self.get_loss(observed_data[0], [observed_data[1],observed_data[2],observed_data[3]], dense_time, tuning_lambda) / observed_data[0].shape[0]
                error = self.get_error(true_pars)        
                curves = self.predict_curves(observed_data[0])
                # when tested dense time
                # curves = self.predict_curves(dense_time)
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




    def __mse_xyz(self, u_true, y_pred):

        # pred = self.predict()

        loss_x = tf.reduce_mean(tf.square(y_pred[0] - u_true[0]))
        loss_y = tf.reduce_mean(tf.square(y_pred[1] - u_true[1]))
        loss_z = tf.reduce_mean(tf.square(y_pred[2] - u_true[2]))

        return (loss_x + loss_y + loss_z)



    def __mse_dzdydz(self, y_pred):

        loss_fx = tf.reduce_mean(tf.square(y_pred[3]))
        loss_fy = tf.reduce_mean(tf.square(y_pred[4]))
        loss_fz = tf.reduce_mean(tf.square(y_pred[5]))

        return loss_fx + loss_fy + loss_fz



