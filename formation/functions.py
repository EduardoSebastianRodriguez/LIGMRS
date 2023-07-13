import torch
import numpy as np
from torch import nn


class ErdosRenyiModel:
    def __init__(self, num_nodes, type, parameters, device):
        self.device = device
        self.num_nodes = num_nodes
        self.type = type
        if self.type == "M":
            self.num_edges = parameters["num_edges"]

        if self.type == "P":
            self.probability = parameters["probability"]

        self.adjacency = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        self.list_of_nodes = np.linspace(0, self.num_nodes - 1, self.num_nodes)

    def generateGraph(self):
        if self.type == "M":
            for i in range(self.num_edges):
                edge = np.random.choice(self.list_of_nodes, 2, replace=False)
                self.addEdge(edge[0], edge[1])

        if self.type == "P":
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if np.random.binomial(1, self.probability):
                        self.addEdge(i, j)

    def removeEdge(self, i, j):
        self.adjacency[i, j] = 0.
        self.adjacency[j, i] = 0.

    def addEdge(self, i, j):
        self.adjacency[i, j] = 1.
        self.adjacency[j, i] = 1.

    def removeNode(self, i):
        self.list_of_nodes = np.concatenate((self.list_of_nodes[:i], self.list_of_nodes[i+1:] - 1))
        self.adjacency = torch.cat((self.adjacency[:i, :], self.adjacency[i+1:, :]), dim=0)
        self.adjacency = torch.cat((self.adjacency[:, :i], self.adjacency[:, i+1:]), dim=1)
        self.num_nodes -= 1

    def addNode(self, i):
        self.list_of_nodes = np.concatenate((self.list_of_nodes[:i], np.array([i]), self.list_of_nodes[i:] + 1))
        self.adjacency = torch.cat((self.adjacency[:i, :], torch.zeros(1, self.num_nodes, device=self.device), self.adjacency[i:, :]), dim=0)
        self.num_nodes += 1
        self.adjacency = torch.cat((self.adjacency[:, :i], torch.zeros(1, self.num_nodes, device=self.device), self.adjacency[:, i:]), dim=1)

    def generateLaplacian(self):
        degree_matrix = torch.diag(torch.sum(self.adjacency, dim=1))
        self.laplacian = degree_matrix - self.adjacency

    def resetGraph(self):
        self.adjacency = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        self.list_of_nodes = np.linspace(0, self.num_nodes - 1, self.num_nodes)
        self.laplacian = torch.zeros_like(self.adjacency, device=self.device)


def nodalDistance(X):
    Xi = X.repeat(X.shape[0], 1)
    Xj = X.repeat_interleave(X.shape[0], 0)
    E = (Xi - Xj).norm(2, dim=1).pow(2)
    return E.reshape(X.shape[0], X.shape[0])

def vectorizedNodalDistance(E, give_s, device):
    num_nodes = E.shape[0]
    indeces = torch.nonzero(torch.triu(E+1, diagonal=1))
    e = E[indeces[:, 0], indeces[:, 1]].reshape(-1, 1)

    if give_s:
        num_cols = e.shape[0]
        I = np.zeros([num_cols, 1])
        J = np.zeros([num_cols, 1])
        k = 0
        for i in range(1, num_nodes):
            I[k:k + (num_nodes - i), 0] = np.around(np.linspace(i, num_nodes - 1, num_nodes - i))
            J[k:k + (num_nodes - i), 0] = np.around(i - 1)
            k = k + (num_nodes - i)
        S = torch.zeros(num_nodes, num_cols, device=device)
        index_i = np.concatenate((I[:, 0], J[:, 0])).astype(int)
        index_j = np.concatenate((np.linspace(0, num_cols - 1, num_cols).astype(int),
                                  np.linspace(0, num_cols - 1, num_cols).astype(int)))
        S[(index_i, index_j)] = 1

        return e, S

    else:
        return e


def training(  input_X,
               input_laplacians,
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
               device):

    """ setup the optimizer and the list to store data """
    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr)
    encoder.train()
    weighted_adjacency_matrices = []
    losses = []

    """ training loop """
    for k in range(num_iter):
        """ initialize the neural network """
        if k == 0:
            X = input_X[:10000, :, :]
            x, alpha, beta = encoder(X[0, :, :].unsqueeze(0).transpose(0, 1))
            E = nodalDistance(x)
            e, S = vectorizedNodalDistance(E, True, device)
            e_bar = torch.clone(e)
            t_k = 1
            lambda_k = torch.rand(num_nodes, 1, device=device)
            w_k = torch.clone(lambda_k)
            laplacians = input_laplacians

        """ set input data as a function of the training step """
        L = (num_nodes - 1) / torch.linalg.norm(beta)
        if k == 0:
            x_in = X[k, :, :].unsqueeze(0).transpose(0, 1)
            norm = 1
        else:
            if k <= window:
                x_in = X[:k, :, :].transpose(0, 1)
                norm = k
            else:
                x_in = X.transpose(0, 1)
                norm = window

        """ encoder """
        x, alpha, beta = encoder(x_in)

        """ optimization module """
        E = nodalDistance(x)
        e = vectorizedNodalDistance(E, False, device) / norm
        e_bar = gamma * e_bar.data + (1 - gamma) * e
        w_bar_k = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                (S.T @ w_k.data - 2 * e_bar) / (2 * torch.linalg.norm(beta)))
        difference = S @ w_bar_k - L.data * w_k.data
        u_k = (difference + torch.sqrt((difference.pow(2)) + 4 * torch.linalg.norm(alpha) * L.data)) / 2
        lambda_k_1 = w_k.data - (1 / L.data) * (S @ w_bar_k.data - u_k)
        t_k_1 = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2
        w_k = lambda_k_1 + ((t_k - 1) / t_k_1) * (lambda_k_1 - lambda_k)
        w_hat_k_1 = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                  (S.T @ lambda_k_1 - 2 * e_bar) / (2 * torch.linalg.norm(beta)))

        lambda_k = lambda_k_1
        t_k = t_k_1

        if k % reset == 0:
            t_k = 1

        """ store the discovered weighted adjacency matrix """
        w_stored = torch.clone(w_bar_k).data
        w_stored[w_stored < threshold] = 0
        weighted_adjacency_matrices.append(w_stored)

        """compute the loss """
        indeces = torch.nonzero(torch.triu(-laplacians + 1, diagonal=1))
        w_gt = -laplacians[indeces[:, 0], indeces[:, 1]]
        loss = torch.abs(w_gt - w_hat_k_1.squeeze(1)).sum() + torch.abs((1 - w_gt) - (1 - w_hat_k_1.squeeze(1))).sum()

        losses.append(loss.item())

        """ backprop """
        optimizer_encoder.zero_grad()

        loss.backward()

        optimizer_encoder.step()

        """ store data """
        if k % store_interval == 0:
            torch.save(encoder, "encoder_"+str(k)+".pth")
            print(f'iteration {k}: ',
                  f'loss is {loss.item():.3f}')

    return weighted_adjacency_matrices, losses


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0], bias=True), activation]
        for i in range(len(self.hidden_channels)-1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i+1], bias=True))
            if i < len(self.hidden_channels)-2:
                layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP_params(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0], bias=True), nn.Tanh()]
        for i in range(len(self.hidden_channels)-1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i+1], bias=True))
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class encoderModel(nn.Module):
    def __init__(self, device, mlp_in_channels, mlp_hidden_channels):
        super().__init__()

        self.mlp = MLP(in_channels=mlp_in_channels, hidden_channels=mlp_hidden_channels, activation=nn.Tanh()).to(device)
        self.mlp_in = MLP_params(in_channels=2, hidden_channels=[2, 1], activation=nn.Tanh()).to(device)
        self.mlp_alpha = MLP_params(in_channels=1, hidden_channels=mlp_hidden_channels, activation=nn.Sigmoid())
        self.mlp_beta = MLP_params(in_channels=1, hidden_channels=mlp_hidden_channels, activation=nn.Sigmoid())
        self.device = device
        self.softmax = nn.Softmax(dim=0)
        self.slope = 5

    def forward(self, x):
        x = self.mlp_in(x.reshape(-1, 2)).reshape(x.shape[0], x.shape[1])
        x = self.softmax(x @ x.T) @ x
        x = self.mlp(x.reshape(-1, 1)).reshape(x.shape)
        e = nodalDistance(x)
        return x, torch.exp(self.slope*self.mlp_alpha(torch.mean(e).unsqueeze(0))), torch.exp(-self.slope*self.mlp_beta(torch.mean(e).unsqueeze(0)))


