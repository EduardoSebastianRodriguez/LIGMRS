import numpy as np
import torch
import matplotlib.pyplot as plt

datasets = [1, 2, 3]
num_nodes = 50
max_num_edges = num_nodes * (num_nodes - 1) / 2
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots()
for i in range(len(datasets)):
    MSE_mean_sota = torch.load("MSE_mean_"+str(True)+"_"+str(datasets[i]))
    MSE_mean = torch.load("MSE_mean_" + str(False) + "_" + str(datasets[i]))
    MSE_std_sota = torch.load("MSE_std_"+str(True)+"_"+str(datasets[i]))
    MSE_std = torch.load("MSE_std_" + str(False) + "_" + str(datasets[i]))
    edges_sota = torch.load("edges_densities_"+str(True)+"_"+str(datasets[i]))
    edges = torch.load("edges_densities_" + str(False) + "_" + str(datasets[i]))

    ax.fill_between(np.array(edges)/max_num_edges, np.array(MSE_mean)/max_num_edges - np.array(MSE_std)/max_num_edges, np.array(MSE_mean)/max_num_edges + np.array(MSE_std)/max_num_edges, color='C' + str(i), alpha=0.3)
    ax.plot(np.array(edges)/max_num_edges, np.array(MSE_mean)/max_num_edges, color='C'+str(i), linewidth=3, marker='o', label='flocking '+str(datasets[i]), markersize=12)
    ax.fill_between(np.array(edges_sota)/max_num_edges, np.array(MSE_mean_sota)/max_num_edges - np.array(MSE_std_sota)/max_num_edges, np.array(MSE_mean_sota)/max_num_edges + np.array(MSE_std_sota)/max_num_edges, color='C' + str(i), alpha=0.3)
    ax.plot(np.array(edges_sota)/max_num_edges, np.array(MSE_mean_sota)/max_num_edges, color='C'+str(i), linewidth=3, marker='d', linestyle='--', markersize=12)
ax.set_xlabel(r"$\rho(\mathcal{G})$")
ax.set_ylabel(r"$MAE$")
ax.legend(markerscale=0)
ax.set_yscale('log')
plt.show()
