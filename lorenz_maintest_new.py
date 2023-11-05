import numpy as np
import tensorflow as tf

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.integrate import odeint
mpl.style.use('seaborn-paper')  
import pandas as pd

from src import LorenzPINN
import scipy.io
import sys

seeds = 3
numofpt = [25, 40, 50]
noise_level = 0.15
noise_YN = True
initial = (5.0, 5.0, 5.0)
# dense = [1]
# dense = [2, 10, 20]
# dense50 = [4, 10, 26]
# dense80 = [3, 7, 10]
# dense100 = [2, 5, 8]
# dense60 = [4]
# dense70 = [3]
# 200, 400, 1000
dense25 = [8, 16, 40]
dense40 = [5, 10, 25]
dense50 = [4, 8, 20]
tuning_lambda = [10]
# srb 10, 28, 8/3
beta = 2.667
rho = 28
sigma = 10
dfchaos = 'groundtruthcurvesobs.csv'
dfstable = 'stablegroundtruthcurvesobs.csv'
stable = False

def lorenz(y0, t, s=sigma, r=rho, b=beta):
    '''
    Given:
    x, y, z: a point of interest in three dimensional space
    s, r, b: parameters defining the lorenz attractor
    Returns:
    x_dot, y_dot, z_dot: values of the lorenz attractor's partial
    derivatives at the point x, y, z
    '''
    x, y, z = y0
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def gen_traindata():
    t = np.linspace(0., 10.0, n)
    y = odeint(func=lorenz, y0=np.array(initial), t=t)
    if noise_YN == True:
        noise = np.ptp(y, axis=0)*noise_level
        y += np.random.normal(loc=0.0, scale=noise, size=(t.shape[0], 3))
    y[0] = np.array(initial)
    return t.reshape(-1,1), y


for seed in range(seeds):
    np.random.seed(seed)
    for n in numofpt:
        loaded_data = gen_traindata()
        if n == 25:
            dense = dense25
        elif n == 40:
            dense = dense40
        elif n == 50:
            dense = dense50
        for d in dense:
            file = open('1103'+"N"+str(n)+str(noise_YN)+"D"+str(d)+"output.txt", "a")
            t_dense = np.linspace(0., 10.0, d*n)
            t_dense = t_dense.reshape(-1,1)
            for l in tuning_lambda:
                def fit_lorenz():
                    pars = [sigma,rho,8/3]
                     # beta initialize
                    lorenz_data = [loaded_data[0], loaded_data[1][:,0], loaded_data[1][:,1], loaded_data[1][:,2]]

                    ts = lorenz_data[0].reshape(-1,1)
                    xs = lorenz_data[1].reshape(-1,1)
                    ys = lorenz_data[2].reshape(-1,1)
                    zs = lorenz_data[3].reshape(-1,1)
                    lorenz_data = [ts, xs, ys, zs]
                    pinn = LorenzPINN(bn=True, log_opt=True, lr=1e-2, layers=3, layer_width=32)
                    output = []
                    for i in range(600):
                        pinn.fit(lorenz_data, pars, 100, t_dense, l, verbose=True)
                        curves = pinn.predict_curves(t_dense)
                        mse = pinn.get_loss(lorenz_data[0], [lorenz_data[1],lorenz_data[2],lorenz_data[3]], t_dense, l) / lorenz_data[0].shape[0]
                        mse = mse.numpy().item()
                        e = pinn.get_error(pars).numpy().item()
                        output.append([(i+1)*100, np.exp(pinn.c3.numpy()), np.exp(pinn.c2.numpy()), np.exp(pinn.c1.numpy()), mse, e])
                        header = ["epoch", "beta", "rho", "sigma", "loss", "error"]
                        pd.DataFrame(np.array(output)).to_csv('new'+str(n)+"N"+str(noise_YN)+"D"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'output.csv', header=header, index=False)
                    # MSE plots
                    output = np.array(output)
                    df = [output[:, 4], output[:, 5]]
                    fig, ax = plt.subplots(1, 2, dpi=200)
                    for i in range(2):
                       ax[i].hist(df[i])
                       ax[i].set_title(['Loss', 'Error'][i])
                    plt.savefig('1103'+str(n)+"N"+str(noise_YN)+"D"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'MSE.png')
                    plt.close()
                
                    fig, ax = plt.subplots(1, 3, dpi=200, figsize=(16, 3))
                    # obs_data = [xs[:,0], ys[:,0], zs[:,0]]
                    if stable == True:
                        csvfile = dfstable
                    else:
                        csvfile = dfchaos
                    obs = pd.read_csv(csvfile)
                    obs = [obs['X'], obs['Y'], obs['Z']]

                    # XYZ plot
                    for i in range(3):
                        ax[i].plot(t_dense, curves[i][:,0], color="red", label="pred.")
                        ax[i].plot(np.linspace(0., 10.0, len(obs[0])).reshape(-1,1), obs[i], color="black", linewidth=1, label="ground truth")
                        ax[i].scatter(ts, lorenz_data[i+1], s=8, label="obs.")
                        ax[i].set_title(["X", "Y", "Z"][i])
                        ax[i].legend()
                    plt.savefig('1103'+str(n)+"N"+str(noise_YN)+"D"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'xyz.png')
                    plt.close()

    
                    print(
                        "Estimated parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
                            np.exp(pinn.c1.numpy().item()), np.exp(pinn.c2.numpy().item()), np.exp(pinn.c3.numpy().item())
                            )  
                    )
                    print('\n')
                    print(
                            "True parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
                                pars[0], pars[1], pars[2]
                            )
                        )   
                    file.write(str(n)+"N"+str(noise_YN)+"D"+str(d)+'lambda'+str(l)+": Estimated parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
                            np.exp(pinn.c1.numpy().item()), np.exp(pinn.c2.numpy().item()), np.exp(pinn.c3.numpy().item()))+"\n")
                       

            if __name__ == "__main__":
                fit_lorenz()
            file.close()


