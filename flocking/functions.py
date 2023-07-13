import torch
import numpy as np
from torch import nn


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
               resample_interval,
               lr,
               device):

    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr)
    encoder.train()
    weighted_adjacency_matrices = []
    weighted_adjacency_matrices_ground_truth = []
    losses = []
    for k in range(num_iter):

        if k % resample_interval == 0:
            counter = np.random.randint(0, int(input_X.shape[0]/10)-1)
            X = input_X[counter*10: (counter+1)*10, :, :]
            x, alpha, beta = encoder(X.transpose(0, 1))
            E = nodalDistance(x)
            e, S = vectorizedNodalDistance(E, True, device)
            e_bar = torch.clone(e)
            t_k = 1
            lambda_k = torch.rand(num_nodes, 1, device=device)
            w_k = torch.clone(lambda_k)
            laplacians = input_laplacians[counter*10: (counter+1)*10].mean(0)

        L = (num_nodes - 1) / torch.linalg.norm(beta)
        norm = window

        x, alpha, beta = encoder(X.transpose(0, 1))
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

        indeces = torch.nonzero(torch.triu(-laplacians + 1, diagonal=1))
        w_gt = -laplacians[indeces[:, 0], indeces[:, 1]]
        adjoint_1 = torch.ones(w_gt.shape[0], device=device)
        adjoint_2 = torch.ones(w_gt.shape[0], device=device)
        loss_1 = torch.abs(w_gt - w_hat_k_1.squeeze(1)).mean()
        loss_2 = torch.abs((adjoint_1 - w_gt) - (adjoint_2 - w_hat_k_1.squeeze(1))).mean()
        loss = loss_1 + loss_2

        optimizer_encoder.zero_grad()

        loss.backward()

        optimizer_encoder.step()

        if k % store_interval == 0:
            losses.append(loss.item())
            w_stored = torch.clone(w_bar_k).data
            w_stored[w_stored < threshold] = 0
            weighted_adjacency_matrices.append(w_stored)
            weighted_adjacency_matrices_ground_truth.append(w_gt)
            torch.save(encoder, "encoder_flocking_"+str(k)+".pth")
            print(f'iteration {k}: ',
                  f'loss1 is {loss_1.item():.3f},',
                  f'loss2 is {loss_2.item():.3f},')

    return weighted_adjacency_matrices, weighted_adjacency_matrices_ground_truth, losses


def evaluation(input_X,
               input_laplacians,
               encoder,
               gamma,
               tol,
               num_nodes,
               num_iter,
               num_graphs,
               num_samples_per_trajectory,
               reset,
               threshold,
               window,
               resample_interval,
               device,
               dataset,
               SOTA):

    weighted_adjacency_matrices = []
    weighted_adjacency_matrices_ground_truth = []
    MSE = []
    MSE_mean = []
    MSE_std = []
    edges_densities = []
    edges_densities_prev = []
    edges_densities_numpy = []
    encoder.eval()
    counter = 0

    alpha = torch.as_tensor([1.0])
    beta = torch.as_tensor([0.00001])

    selector = 0
    www = torch.zeros(4, int(num_nodes*(num_nodes-1)/2), device=device)
    if SOTA:
        with torch.no_grad():
            for k in range(num_graphs * num_iter):
                if k % num_iter == 0:
                    X = input_X[dataset-1][counter, :, :].unsqueeze(0).repeat(10, 1, 1)
                    laplacians = input_laplacians[dataset-1][counter]
                    counter += 1
                    selector = 0
                if k % resample_interval == 0:
                    x = X[:, :, selector].transpose(0, 1)
                    E = nodalDistance(x)
                    e, S = vectorizedNodalDistance(E, True, device)
                    e_bar = torch.clone(e)
                    t_k = 1
                    lambda_k = torch.rand(num_nodes, 1, device=device)
                    w_k = torch.clone(lambda_k)
                    if selector == 1:
                        selector = 0
                    else:
                        selector = 1

                L = (num_nodes - 1) / torch.linalg.norm(beta)
                norm = window

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

                lambda_k = lambda_k_1
                t_k = t_k_1

                if k % reset == 0:
                    t_k = 1

                indeces = torch.nonzero(torch.triu(-laplacians + 1, diagonal=1))
                w_gt = -laplacians[indeces[:, 0], indeces[:, 1]]

                if (k+1) % resample_interval == 0:
                    www[selector, :] = torch.clone(w_bar_k.reshape(-1))

                if (k+1) % num_iter == 0:
                    w_stored = torch.clone(www.mean(dim=0)).data
                    w_stored[w_stored < threshold] = 0
                    mse = (w_stored.reshape(-1) - w_gt.reshape(-1)).pow(2).pow(0.5).sum()
                    MSE.append(mse.detach().cpu().numpy())
                    edges_ground_truth_bool = w_gt.reshape(-1).bool()
                    density = torch.sum((edges_ground_truth_bool.int()))
                    edges_densities_prev.append(density.detach().cpu().numpy())

                if (k+1) % (num_iter * num_samples_per_trajectory) == 0:
                    MSE_mean.append(np.mean(np.array(MSE)))
                    MSE_std.append(np.std(np.array(MSE)))
                    MSE = []
                    edges_densities_numpy.append(np.mean(np.array(edges_densities_prev)))
                    edges_densities_prev = []

                if (k + 1 + num_iter) % (num_iter * num_samples_per_trajectory) == 0:
                    weighted_adjacency_matrices.append(w_stored)
                    weighted_adjacency_matrices_ground_truth.append(w_gt)
                    edges_densities.append(density)
                    print(f'iteration {k}')
    else:
        with torch.no_grad():
            for k in range(num_graphs * num_iter):
                if k % num_iter == 0:
                    X = input_X[dataset-1][counter, :, :].unsqueeze(0).repeat(10, 1, 1)
                    x, alpha, beta = encoder(X.transpose(0, 1))
                    E = nodalDistance(x)
                    e, S = vectorizedNodalDistance(E, True, device)
                    e_bar = torch.clone(e)
                    t_k = 1
                    lambda_k = torch.rand(num_nodes, 1, device=device)
                    w_k = torch.clone(lambda_k)
                    laplacians = input_laplacians[dataset-1][counter]
                    counter += 1

                L = (num_nodes - 1) / torch.linalg.norm(beta)
                norm = window

                x, alpha, beta = encoder(X.transpose(0, 1))
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

                lambda_k = lambda_k_1
                t_k = t_k_1

                if k % reset == 0:
                    t_k = 1

                indeces = torch.nonzero(torch.triu(-laplacians + 1, diagonal=1))
                w_gt = -laplacians[indeces[:, 0], indeces[:, 1]]

                if (k+1) % num_iter == 0:
                    w_stored = torch.clone(w_bar_k).data
                    w_stored[w_stored < threshold] = 0
                    mse = (w_stored.reshape(-1) - w_gt.reshape(-1)).pow(2).pow(0.5).sum()
                    MSE.append(mse.detach().cpu().numpy())
                    edges_ground_truth_bool = w_gt.reshape(-1).bool()
                    density = torch.sum((edges_ground_truth_bool.int()))
                    edges_densities_prev.append(density.detach().cpu().numpy())

                if (k+1) % (num_iter * num_samples_per_trajectory) == 0:
                    MSE_mean.append(np.mean(np.array(MSE)))
                    MSE_std.append(np.std(np.array(MSE)))
                    MSE = []
                    edges_densities_numpy.append(np.mean(np.array(edges_densities_prev)))
                    edges_densities_prev = []

                if (k + 1 + num_iter) % (num_iter * num_samples_per_trajectory) == 0:
                    weighted_adjacency_matrices.append(w_stored)
                    weighted_adjacency_matrices_ground_truth.append(w_gt)
                    print(f'iteration {k}')

    return weighted_adjacency_matrices, weighted_adjacency_matrices_ground_truth, edges_densities, MSE_mean, MSE_std, edges_densities_numpy


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


class encoderModelFlocking(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.mlp_in = MLP_params(in_channels=2, hidden_channels=[4, 4, 1], activation=nn.Tanh()).to(device)
        self.mlp_alpha = MLP_params(in_channels=1, hidden_channels=[2, 1], activation=nn.Tanh())
        self.mlp_beta = MLP_params(in_channels=1, hidden_channels=[2, 1], activation=nn.Tanh())
        self.device = device
        self.softmax = nn.Softmax(dim=0)
        self.slope = 3

    def forward(self, x):
        x = self.mlp_in(x.reshape(-1, 2)).reshape(x.shape[0], x.shape[1])
        x = self.softmax(x @ x.T) @ x
        e = nodalDistance(x)
        return x, torch.exp(self.slope*self.mlp_alpha(torch.mean(e).unsqueeze(0))), torch.exp(-self.slope*self.mlp_beta(torch.mean(e).unsqueeze(0)))

