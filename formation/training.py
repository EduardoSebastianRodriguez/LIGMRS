import numpy as np
import torch
from functions import ErdosRenyiModel, encoderModel, training

""" parameters and hyperparameters """
# random seed
seed = 23
# device
device = "cpu" if not torch.has_cuda else "cuda"
# erdos-renyi graph parameters
probability = 0.2
num_signals = 20000
num_nodes = 50
sigma_e = 0.1
eps = 1e-3
# encoder
encoder = encoderModel(device=device,
                       mlp_in_channels=1,
                       mlp_hidden_channels=[5, 10, 5, 1]
                       ).to(device)
# graph optimizer parameters
gamma = 0.99
tol = 0
threshold = 1e-5
reset = 50
window = 10000
# training parameters
lr = 1e-3
num_iter = 10000
store_interval = 1000

""" set seeds """
np.random.seed(seed)
torch.manual_seed(seed)

""" generate formation with a an erdos-renyi graph topology"""
erdos_renyi_graph = ErdosRenyiModel(num_nodes, "P", {"probability": probability}, device)
erdos_renyi_graph.resetGraph()
erdos_renyi_graph.generateGraph()
erdos_renyi_graph.generateLaplacian()
D, V = torch.linalg.eig(erdos_renyi_graph.laplacian)
d = torch.linalg.pinv(torch.diag(D + eps).float())
mu = torch.zeros(num_nodes, device=device)

X1 = torch.distributions.MultivariateNormal(mu, d).sample(sample_shape=[int(num_signals)])
X1 = V.float() @ X1.T
X1 = X1 + (sigma_e ** 2) * torch.randn(X1.shape, device=device)
mean, std = torch.mean(X1), torch.std(X1)
X1 = (X1 - mean) / std

X2 = torch.distributions.MultivariateNormal(mu, d).sample(sample_shape=[int(num_signals)])
X2 = V.float() @ X2.T
X2 = X2 + (sigma_e ** 2) * torch.randn(X2.shape, device=device)
mean, std = torch.mean(X2), torch.std(X2)
X2 = (X2 - mean) / std

X1 = X1.T.unsqueeze(dim=1)
X2 = X2.T.unsqueeze(dim=1)
X = torch.cat((X1, X2), dim=1).transpose(1, 2)

""" train the neural network """
edges_history, loss_history = training(X,
                                       erdos_renyi_graph.laplacian,
                                       encoder,
                                       gamma,
                                       tol,
                                       num_nodes,
                                       num_iter,
                                       reset,
                                       threshold,
                                       window,
                                       store_interval,
                                       lr,
                                       device)
torch.save(loss_history, "loss_history")
