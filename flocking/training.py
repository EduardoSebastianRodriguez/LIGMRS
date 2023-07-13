import numpy as np
import torch
from functions import encoderModelFlocking, training

""" parameters and hyperparameters """
# random seed
seed = 23
# device
device = "cpu" if not torch.has_cuda else "cuda"
# flocking parameters
d = torch.as_tensor(0.6)
r = torch.as_tensor(1.2)
numSamples = 150
seed_flocking = 2
numAgents = 50
# encoder
encoder = encoderModelFlocking(device=device).to(device)
# graph optimizer parameters
gamma = 0.99
tol = 0
threshold = 1e-5
reset = 50
window = 10
# training parameters
lr = 1e-3
num_iter = 150000
store_interval = 1000
resample_interval = 500

""" set seeds """
np.random.seed(23)
torch.manual_seed(23)

""" load and preprocess flocking data """
X = torch.load('flocking_trajectories'+str(numAgents)+str(numSamples)+str(seed_flocking)+str(d)+str(r)+'_.pth').to(device)
laplacians = torch.load('flocking_laplacians'+str(numAgents)+str(numSamples)+str(seed_flocking)+str(d)+str(r)+'_.pth')
laplacians = torch.stack(laplacians).to(device)
X = torch.cat((X[:, 0*numAgents:2*numAgents:2].unsqueeze(2), X[:, 0*numAgents + 1:2*numAgents:2].unsqueeze(2)), dim=2)
mean0, std0 = torch.mean(X[:, :, 0]), torch.std(X[:, :, 0])
X[:, :, 0] = (X[:, :, 0] - mean0) / std0
mean1, std1 = torch.mean(X[:, :, 1]), torch.std(X[:, :, 1])
X[:, :, 1] = (X[:, :, 1] - mean1) / std1

""" train the neural network """
edges_history, edges_history_gt, loss_history = training(X,
                                                         laplacians,
                                                         encoder,
                                                         gamma,
                                                         tol,
                                                         numAgents,
                                                         num_iter,
                                                         reset,
                                                         threshold,
                                                         window,
                                                         store_interval,
                                                         resample_interval,
                                                         lr,
                                                         device)
torch.save(loss_history, "loss_history")
