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

seeds = 30
numofpt = [10, 20]
noise_level = 0.15
dense_obs = True
noise_YN = True
initial = (5.0, 5.0, 5.0)
# dense = []
dense10 = [50, 100]
dense20 = [25, 50, 100]
tuning_lambda = [10]
# srb 10, 28, 8/3
beta = 2.667
rho = 28
sigma = 10
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

# generate dense observations
def dense_ob(n, y, d):
    t = np.linspace(0., 10.0, n+1)
    times = [np.linspace(t[i],t[i+1], d) for i in range(n)]
    y_dense = odeint(func=lorenz, y0=np.array(y[0]), t=times[0])
    for i in range(n-1):
        y_temp = odeint(func=lorenz, y0=np.array(y[i+1]), t=times[i+1])
        y_dense = np.concatenate((y_dense, y_temp), axis=0)
    t = np. concatenate(times)
    return t.reshape(-1,1), y_dense

for seed in range(seeds):
    np.random.seed(seed)
    for n in numofpt:
        t1 = np.linspace(0., 10.0, n+1)
        y1 = odeint(func=lorenz, y0=np.array(initial), t=t1)

        # add noise
        noise = np.ptp(y1, axis=0)*noise_level
        y1 += np.random.normal(loc=0.0, scale=noise, size=(t1.shape[0], 3))
        # as the model is sensitive to initial
        y1[0] = np.array(initial)
        t1 = t1.reshape(-1,1)

        # data for plots
        data = [t1, y1[:,0], y1[:,1], y1[:,2]]
        data = [data[0].reshape(-1,1), data[1].reshape(-1,1), data[2].reshape(-1,1), data[3].reshape(-1,1)]
        obsdata = [data[1], data[2], data[3]]
        if n == 20:
            dense = dense20
        if n == 10:
            dense = dense10
        for d in dense:
            file = open('1018'+"N"+str(n)+str(noise_YN)+"D"+str(d)+"output.txt", "a")
            loaded_data = dense_ob(n, y1, d)
            for l in tuning_lambda:
                def fit_lorenz():
                    pars = [10,28,8/3]
                    # beta initialize
                    lorenz_data = [loaded_data[0], loaded_data[1][:,0], loaded_data[1][:,1], loaded_data[1][:,2]]

                    ts = lorenz_data[0].reshape(-1,1)
                    xs = lorenz_data[1].reshape(-1,1)
                    ys = lorenz_data[2].reshape(-1,1)
                    zs = lorenz_data[3].reshape(-1,1)

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(xs[:,0], ys[:,0], zs[:,0], lw=.75)
                    ax.set_xlabel("X Axis", fontsize=16)
                    ax.set_ylabel("Y Axis", fontsize=16)
                    ax.set_zlabel("Z Axis", fontsize=16)
                    ax.set_title("Ground-Truth", fontsize=16)
                    ax.tick_params(axis='both', which='major', labelsize=10,pad=2)
                    ax.tick_params(axis='both', which='minor', labelsize=8,pad=2)
                    plt.savefig('ground_truth.png')
                    plt.close()
    
    
                    lorenz_data = [ts, xs, ys, zs]
        
                    pinn = LorenzPINN(bn=True, log_opt=True, lr=1e-2, layers=3, layer_width=32)
                    output = []
                    for i in range(600):
                        pinn.fit(lorenz_data, pars, 100, l, verbose=True)
                        curves = pinn.predict_curves(lorenz_data[0])
                        mse = pinn.get_loss(lorenz_data[0], [lorenz_data[1],lorenz_data[2],lorenz_data[3]], l) / lorenz_data[0].shape[0]
                        mse = mse.numpy().item()
                        e = pinn.get_error(pars).numpy().item()
                        output.append([(i+1)*100, np.exp(pinn.c3.numpy()), np.exp(pinn.c2.numpy()), np.exp(pinn.c1.numpy()), mse, e])

                        # output += list(np.abs(loaded_data[1] - xhat).mean(axis=0) / np.abs(loaded_data[1]).mean(axis=0))
                        # output += list(np.sqrt(((loaded_data[1] - xhat) ** 2).mean(axis=0)) / np.sqrt((loaded_data[1] ** 2).mean(axis=0)))
                    header = ["epoch", "beta", "rho", "sigma", "loss", "error"]
                    pd.DataFrame(np.array(output)).to_csv('1018'+str(n)+"ND"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'output.csv', header=header, index=False)
                    
                    # MSE plots
                    output = np.array(output)
                    df = [output[:, 4], output[:, 5]]
                    fig, ax = plt.subplots(1, 2, dpi=200)
                    for i in range(2):
                        ax[i].hist(df[i])
                        ax[i].set_title(['Loss', 'Error'][i])
                    plt.savefig('1018'+str(n)+"ND"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'MSE.png')
                    plt.close()
                    
                    # refer code from jupyter notebook
                    fig, ax = plt.subplots(1, 3, dpi=200, figsize=(16, 3))
                    # obs_data = [xs[:,0], ys[:,0], zs[:,0]]
                    obs = pd.read_csv('groundtruthcurvesobs.csv')
                    obs = [obs['X'], obs['Y'], obs['Z']]
                    # for dense observations
                    # if dense_obs==True:
                    #     t1 = np.linspace(0., 10.0, n)
                    #     y1 = odeint(func=lorenz, y0=np.array(initial), t=t1)
                    #     if noise_YN == True:
                    #         noise = np.ptp(y1, axis=0)*noise_level
                    #         # add noise
                    #         y1 += np.random.normal(loc=0.0, scale=noise, size=(t1.shape[0], 3))
                    #     t1 = t1.reshape(-1,1)
                    #     data = [t1, y1[:,0], y1[:,1], y1[:,2]]
                    #     data = [data[0].reshape(-1,1), data[1].reshape(-1,1), data[2].reshape(-1,1), data[3].reshape(-1,1)]
                    #     obsdata = [data[1], data[2], data[3]]
                    # else:
                    #     t1 = ts
                    # (pd.DataFrame(np.array([obsdata[0],obsdata[1],obsdata[2]]), 
                    #              index=["X", "Y", "Z"]).T).to_csv('groundtruthcurvesobs.csv', index=False)


                    # XYZ plot
                    for i in range(3):
                        ax[i].plot(ts, curves[i][:,0], color="red", label="pred.")
                        ax[i].plot(np.linspace(0., 10.0, 1500).reshape(-1,1), obs[i], color="black", linewidth=1, label="ground truth")
                        ax[i].scatter(t1, obsdata[i], s=8, label="obs.")
                        ax[i].set_title(["X", "Y", "Z"][i])
                        ax[i].legend()
                    plt.savefig('1018'+str(n)+"ND"+str(d)+"rand"+str(seed+1)+'lambda'+str(l)+'xyz.png')
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
                    file.write(str(n)+"N"+str(noise_YN)+"D"+str(dense_obs)+'lambda'+str(tuning_lambda)+": Estimated parameters: c1 = {:3.2f}, c2 = {:3.2f}, c3 = {:3.2f}".format(
                            np.exp(pinn.c1.numpy().item()), np.exp(pinn.c2.numpy().item()), np.exp(pinn.c3.numpy().item()))+"\n")
                       

            if __name__ == "__main__":
                fit_lorenz()
            file.close()