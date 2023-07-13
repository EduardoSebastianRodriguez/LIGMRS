import numpy as np
import torch
import matplotlib.pyplot as plt

probability = [0.1, 0.2, 0.4]
num_nodes = np.array([10, 20, 30, 40, 50, 60])
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots()
for i in range(len(probability)):
    MSE_means = torch.load("MSE_means_"+str(probability[i]))
    MSE_stds = torch.load("MSE_stds_"+str(probability[i]))
    MSE_means_comp = torch.load("MSE_means_comp_"+str(probability[i]))
    MSE_stds_comp = torch.load("MSE_stds_comp_"+str(probability[i]))

    ax.fill_between(num_nodes, (np.array(MSE_means) - np.array(MSE_stds))/(num_nodes**2), (np.array(MSE_means) + np.array(MSE_stds))/(num_nodes**2), color='C'+str(i), alpha=0.3)
    ax.plot(num_nodes, np.array(MSE_means)/(num_nodes**2), color='C'+str(i), linewidth=3, marker='o', label=r'$p=$'+str(probability[i]), markersize=12)
    ax.fill_between(num_nodes, (np.array(MSE_means_comp) - np.array(MSE_stds_comp))/(num_nodes**2), (np.array(MSE_means_comp) + np.array(MSE_stds_comp))/(num_nodes**2), color='C'+str(i), alpha=0.3)
    ax.plot(num_nodes, np.array(MSE_means_comp)/(num_nodes**2), color='C'+str(i), linewidth=3, marker='d', linestyle='--', markersize=12)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$MAE$")
ax.legend(markerscale=0)
ax.set_yscale('log')
plt.show()
