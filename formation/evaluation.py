import numpy as np
import torch
from functions import ErdosRenyiModel, nodalDistance, vectorizedNodalDistance
import matplotlib.pyplot as plt
from matplotlib import colors

""" parameters and hyperparameters """
# random seed
seed = 23
# device
device = "cpu" if not torch.has_cuda else "cuda"
# erdos-renyi graph parameters
probability = 0.2
num_signals = 10000
num_nodes = [10, 20, 30, 40, 50, 60]
sigma_e = 0.1
eps = 1e-3
# encoder
encoder = torch.load("encoder_10000.pth")
encoder.eval()
# graph optimizer parameters
gamma = 0.99
tol = 0
threshold = 1e-5
reset = 50
window = 10000
# for the state-of-the-art (sota)
alpha = torch.Tensor([0.3]).to(device)
beta = torch.Tensor([0.0001]).to(device)
# evaluation parameters
store_interval = 100
num_iter = 2000
num_experiments = 20
plot_graphs = True

""" set seeds """
np.random.seed(23)
torch.manual_seed(23)

""" log data """
MSE_means_sota = []
MSE_stds_sota = []
MSE_means_ours = []
MSE_stds_ours = []

""" evaluation loop """
for nodes in range(len(num_nodes)):
    MSE_sota = []
    MSE_ours = []
    for i in range(num_experiments):
        """ for each experiment and number of nodes, generate a new formation topology """
        erdos_renyi_graph = ErdosRenyiModel(num_nodes[nodes], "P", {"probability": probability}, device)
        erdos_renyi_graph.resetGraph()
        erdos_renyi_graph.generateGraph()
        erdos_renyi_graph.generateLaplacian()
        D, V = torch.linalg.eig(erdos_renyi_graph.laplacian)
        d = torch.linalg.pinv(torch.diag(D + eps).float())
        mu = torch.zeros(num_nodes[nodes], device=device)

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

        x1 = X1.T.unsqueeze(dim=1)
        x2 = X2.T.unsqueeze(dim=1)
        X = torch.cat((x1, x2), dim=1).transpose(1, 2)

        """ for the state-of-the-art, run the algorithm for each state dimension """
        x = X1
        E = nodalDistance(x)
        e, S = vectorizedNodalDistance(E, True, device)
        e_bar = torch.clone(e)
        t_k = 1
        lambda_k = torch.rand(num_nodes[nodes], 1, device=device)
        w_k = torch.clone(lambda_k)

        for k in range(num_iter):

            L = (num_nodes[nodes] - 1) / beta
            norm = window

            with torch.no_grad():
                x = X1
                E = nodalDistance(x)
                e = vectorizedNodalDistance(E, False, device) / norm
                e_bar = gamma * e_bar.data + (1 - gamma) * e
                w_bar_k = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                             (S.T @ w_k.data - 2 * e_bar) / (2 * beta))
                difference = S @ w_bar_k - L.data * w_k.data
                u_k = (difference + torch.sqrt((difference.pow(2)) + 4 * alpha * L.data)) / 2
                lambda_k_1 = w_k.data - (1 / L.data) * (S @  w_bar_k.data - u_k)
                t_k_1 = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2
                w_k = lambda_k_1 + ((t_k - 1) / t_k_1) * (lambda_k_1 - lambda_k)
                w_hat_k_1 = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                          (S.T @ lambda_k_1 - 2 * e_bar) / (2 * beta))

                lambda_k = lambda_k_1
                t_k = t_k_1

                if k % reset == 0:
                    t_k = 1

                w_stored_1 = torch.clone(w_bar_k).data
                w_stored_1[w_stored_1 < threshold] = 0

                if k % 100 == 0:
                    print(f'SOTA, dimension 1, experiment {i}, iteration {k}')

        x = X2
        E = nodalDistance(x)
        e, S = vectorizedNodalDistance(E, True, device)
        e_bar = torch.clone(e)
        t_k = 1
        lambda_k = torch.rand(num_nodes[nodes], 1, device=device)
        w_k = torch.clone(lambda_k)

        for k in range(num_iter):

            L = (num_nodes[nodes] - 1) / beta
            norm = window

            with torch.no_grad():
                x = X2
                E = nodalDistance(x)
                e = vectorizedNodalDistance(E, False, device) / norm
                e_bar = gamma * e_bar.data + (1 - gamma) * e
                w_bar_k = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                             (S.T @ w_k.data - 2 * e_bar) / (2 * beta))
                difference = S @ w_bar_k - L.data * w_k.data
                u_k = (difference + torch.sqrt((difference.pow(2)) + 4 * alpha * L.data)) / 2
                lambda_k_1 = w_k.data - (1 / L.data) * (S @  w_bar_k.data - u_k)
                t_k_1 = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2
                w_k = lambda_k_1 + ((t_k - 1) / t_k_1) * (lambda_k_1 - lambda_k)
                w_hat_k_1 = torch.maximum(tol * torch.ones_like(e_bar, device=device),
                                          (S.T @ lambda_k_1 - 2 * e_bar) / (2 * beta))

                lambda_k = lambda_k_1
                t_k = t_k_1

                if k % reset == 0:
                    t_k = 1

                w_stored_2 = torch.clone(w_bar_k).data
                w_stored_2[w_stored_2 < threshold] = 0

                if k % store_interval == 0:
                    print(f'SOTA, dimension 2, experiment {i}, iteration {k}')

        mse = (((w_stored_1+w_stored_2)/2).reshape(-1) - erdos_renyi_graph.adjacency[np.nonzero(np.triu(np.ones([num_nodes[nodes], num_nodes[nodes]]), k=1))].reshape(-1)).pow(2).pow(0.5).sum()
        MSE_sota.append(mse.detach().cpu().numpy())

        """ our algorithm """
        x, alpha, beta = encoder(X.transpose(0, 1))
        E = nodalDistance(x)
        e, S = vectorizedNodalDistance(E, True, device)
        e_bar = torch.clone(e)
        t_k = 1
        lambda_k = torch.rand(num_nodes[nodes], 1, device=device)
        w_k = torch.clone(lambda_k)
        norm = window

        with torch.no_grad():
            for k in range(num_iter):
                x, alpha, beta = encoder(X.transpose(0, 1))
                L = (num_nodes[nodes] - 1) / torch.linalg.norm(beta)
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

                w_stored = torch.clone(w_bar_k).data
                w_stored[w_stored < threshold] = 0

                if k % store_interval == 0:
                    print(f'OURS, experiment {i}, iteration {k}')

        upper_triangular = torch.zeros(num_nodes[nodes], num_nodes[nodes], device=device)
        upper_triangular[np.nonzero(np.triu(np.ones([num_nodes[nodes], num_nodes[nodes]]), k=1))] = w_stored.reshape(-1)
        discovered_adjacency = upper_triangular + upper_triangular.T
        discovered_adjacency_full = torch.clone(discovered_adjacency)
        indeces = torch.nonzero(discovered_adjacency)
        discovered_adjacency[indeces[:, 0], indeces[:, 1]] = 1

        mse = (w_stored.reshape(-1) - erdos_renyi_graph.adjacency[np.nonzero(np.triu(np.ones([num_nodes[nodes], num_nodes[nodes]]), k=1))].reshape(-1)).pow(2).pow(0.5).sum()
        MSE_ours.append(mse.detach().cpu().numpy())

        if i == 0 and plot_graphs:
            matrix = erdos_renyi_graph.adjacency.detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='viridis', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            matrix = discovered_adjacency_full.detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='viridis', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            matrix = (discovered_adjacency - erdos_renyi_graph.adjacency).detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='bwr', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

            matrix = (discovered_adjacency_full - erdos_renyi_graph.adjacency).detach().cpu().numpy()
            maximum = np.maximum(np.abs(matrix.max()), np.abs(matrix.min()))
            fig, ax = plt.subplots()
            pcm = ax.imshow(matrix, cmap='bwr', norm=colors.Normalize(vmin=-maximum, vmax=maximum))
            fig.colorbar(pcm, ax=ax, location='bottom', orientation='horizontal', shrink=0.5, pad=0.01)

            # draw gridlines
            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            plt.show()

    MSE_mean_sota = np.mean(MSE_sota)
    MSE_std_sota = np.std(MSE_sota)
    MSE_means_sota.append(MSE_mean_sota)
    MSE_stds_sota.append(MSE_std_sota)

    MSE_mean_ours = np.mean(MSE_ours)
    MSE_std_ours = np.std(MSE_ours)
    MSE_means_ours.append(MSE_mean_ours)
    MSE_stds_ours.append(MSE_std_ours)

    print(f'Results are: MSE SOTA = {MSE_mean_sota}|{MSE_std_sota}', f' MSE ours = {MSE_mean_ours}|{MSE_std_ours}')

torch.save(MSE_means_sota, "MSE_means_comp_"+str(probability))
torch.save(MSE_stds_sota, "MSE_stds_comp_"+str(probability))

torch.save(MSE_means_ours, "MSE_means_"+str(probability))
torch.save(MSE_stds_ours, "MSE_stds_"+str(probability))

