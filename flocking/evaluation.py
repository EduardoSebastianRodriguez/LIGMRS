import numpy as np
import matplotlib.pyplot as plt
import torch
from functions import encoderModelFlocking, evaluation
from matplotlib import colors

""" evaluation parameters """
dataset = 1 # This goes from 1 to 3
SOTA = False # If False, our approached is used; if True, the state-of-the-art is used
plot_graphs = True # If True, some qualitative results are displayed

""" parameters and hyperparameters """
# random seed
seed = 23
# device
device = "cpu" if not torch.has_cuda else "cuda"
# flocking parameters
numSamples = 150
num_graphs = numSamples - 1
num_samples_per_trajectory = 10
seed_flocking = [2, 23, 2, 10]
numAgents = 50
# encoder
encoder = torch.load('encoder_flocking_130000')
# graph optimizer parameters
gamma = 0.99
tol = 0
threshold = 1e-5
reset = 50
window = 10
# evaluation parameters
num_iter = 2000
resample_interval = 500

""" set seeds """
np.random.seed(23)
torch.manual_seed(23)

""" load flocking trajectories """
d1 = torch.as_tensor(0.7)
r1 = torch.as_tensor(1.2)
X1 = torch.load('flocking_trajectories'+str(numAgents)+str(numSamples)+str(seed_flocking[1])+str(d1)+str(r1)+'_.pth').to(device)
laplacians1 = torch.load('flocking_laplacians'+str(numAgents)+str(numSamples)+str(seed_flocking[1])+str(d1)+str(r1)+'_.pth')
laplacians1 = torch.stack(laplacians1).to(device)
X1 = torch.cat((X1[:, 0*numAgents:2*numAgents:2].unsqueeze(2), X1[:, 0*numAgents + 1:2*numAgents:2].unsqueeze(2)), dim=2)
mean0, std0 = torch.mean(X1[:, :, 0]), torch.std(X1[:, :, 0])
X1[:, :, 0] = (X1[:, :, 0] - mean0) / std0
mean1, std1 = torch.mean(X1[:, :, 1]), torch.std(X1[:, :, 1])
X1[:, :, 1] = (X1[:, :, 1] - mean1) / std1

d2 = torch.as_tensor(1.0)
r2 = torch.as_tensor(1.2)
X2 = torch.load('flocking_trajectories'+str(numAgents)+str(numSamples)+str(seed_flocking[2])+str(d2)+str(r2)+'_.pth').to(device)
laplacians2 = torch.load('flocking_laplacians'+str(numAgents)+str(numSamples)+str(seed_flocking[2])+str(d2)+str(r2)+'_.pth')
laplacians2 = torch.stack(laplacians2).to(device)
X2 = torch.cat((X2[:, 0*numAgents:2*numAgents:2].unsqueeze(2), X2[:, 0*numAgents + 1:2*numAgents:2].unsqueeze(2)), dim=2)
mean0, std0 = torch.mean(X2[:, :, 0]), torch.std(X2[:, :, 0])
X2[:, :, 0] = (X2[:, :, 0] - mean0) / std0
mean1, std1 = torch.mean(X2[:, :, 1]), torch.std(X2[:, :, 1])
X2[:, :, 1] = (X2[:, :, 1] - mean1) / std1

d3 = torch.as_tensor(0.6)
r3 = torch.as_tensor(1.2)
X3 = torch.load('flocking_trajectories'+str(numAgents)+str(numSamples)+str(seed_flocking[3])+str(d3)+str(r3)+'_.pth').to(device)
laplacians3 = torch.load('flocking_laplacians'+str(numAgents)+str(numSamples)+str(seed_flocking[3])+str(d3)+str(r3)+'_.pth')
laplacians3 = torch.stack(laplacians3).to(device)
X3 = torch.cat((X3[:, 0*numAgents:2*numAgents:2].unsqueeze(2), X3[:, 0*numAgents + 1:2*numAgents:2].unsqueeze(2)), dim=2)
mean0, std0 = torch.mean(X3[:, :, 0]), torch.std(X3[:, :, 0])
X3[:, :, 0] = (X3[:, :, 0] - mean0) / std0
mean1, std1 = torch.mean(X3[:, :, 1]), torch.std(X3[:, :, 1])
X3[:, :, 1] = (X3[:, :, 1] - mean1) / std1


edges_history, edges_history_gt, edges_densities, MSE_mean, MSE_std, edges_densities_numpy = evaluation([X1, X2, X3],
                                                                                                        [laplacians1, laplacians2, laplacians3],
                                                                                                        encoder,
                                                                                                        gamma,
                                                                                                        tol,
                                                                                                        numAgents,
                                                                                                        num_iter,
                                                                                                        num_graphs,
                                                                                                        num_samples_per_trajectory,
                                                                                                        reset,
                                                                                                        threshold,
                                                                                                        window,
                                                                                                        resample_interval,
                                                                                                        device,
                                                                                                        dataset,
                                                                                                        SOTA)
torch.save(MSE_mean, "MSE_mean_"+str(SOTA)+"_"+str(dataset))
torch.save(MSE_std, "MSE_std_"+str(SOTA)+"_"+str(dataset))
torch.save(edges_densities_numpy, "edges_densities_"+str(SOTA)+"_"+str(dataset))

plt.rcParams.update({'font.size': 22})

if plot_graphs:
    for i in range(int(numSamples/10)):
        if i % 2 == 0:
            upper_triangular = torch.zeros(numAgents, numAgents, device=device)
            upper_triangular[np.nonzero(np.triu(np.ones([numAgents, numAgents]), k=1))] = edges_history[i].reshape(-1)
            discovered_adjacency = upper_triangular + upper_triangular.T
            discovered_adjacency_full = torch.clone(discovered_adjacency)
            indeces = torch.nonzero(discovered_adjacency)
            discovered_adjacency[indeces[:, 0], indeces[:, 1]] = 1

            upper_triangular = torch.zeros(numAgents, numAgents, device=device)
            upper_triangular[np.nonzero(np.triu(np.ones([numAgents, numAgents]), k=1))] = edges_history_gt[i].reshape(-1)
            gt_adjacency = upper_triangular + upper_triangular.T
            gt_adjacency_full = torch.clone(gt_adjacency)
            indeces = torch.nonzero(gt_adjacency)
            gt_adjacency[indeces[:, 0], indeces[:, 1]] = 1

            matrix1 = gt_adjacency_full.detach().cpu().numpy()
            matrix2 = discovered_adjacency_full.detach().cpu().numpy()
            maximum1 = np.maximum(np.abs(matrix1.max()), np.abs(matrix1.min()))
            maximum2 = np.maximum(np.abs(matrix2.max()), np.abs(matrix2.min()))

            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix1, cmap='viridis', norm=colors.Normalize(vmin=-maximum1, vmax=maximum1))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix2, cmap='viridis', norm=colors.Normalize(vmin=-maximum2, vmax=maximum2))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            matrix = (discovered_adjacency - gt_adjacency).detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='bwr', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            matrix = (discovered_adjacency_full - gt_adjacency_full).detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='bwr', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()
