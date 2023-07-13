import torch
from torchdiffeq import odeint


class realSystem():

    def __init__(self, parameters):

        self.na     = parameters['na']
        self.d      = parameters['d']
        self.r      = parameters['r']
        self.e      = parameters['e']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
        self.ha     = parameters['ha']
        self.c1     = parameters['c1']
        self.c2     = parameters['c2']
        self.noise  = parameters['noise']
        self.TV = parameters["TV"]
        self.vx = 1*torch.rand(self.na)
        self.A = 1*torch.rand(self.na)
        self.w = 1*torch.rand(self.na)
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.phi = 2*self.pi*torch.rand(self.na)
        self.save_laplacians = []

    def sigma_norm(self, z, e):
        return (torch.sqrt(1 + e * z.norm(p=2) ** 2) - 1) / e

    def rho_function(self, z, h):
        if 0 <= z < h:
            return 1
        elif h <= z <= 1:
            pi = torch.acos(torch.zeros(1)).item() * 2
            return (1 + torch.cos(pi * ((z - h) / (1 - h)))) / 2
        else:
            return 0

    def sigmoid(self, z):
        return z / (torch.sqrt(1 + z ** 2))

    def phi_function(self, z, a, b, c):
        return ((a + b) * self.sigmoid(z + c) + (a - b)) / 2

    def phi_alpha_function(self, z, r, h, d, a, b, c):
        return self.rho_function(z / r, h) * self.phi_function(z - d, a, b, c)

    def f_control(self, q_agents, p_agents, q_dynamic, p_dynamic):
        return -self.c1 * (q_agents - q_dynamic) - self.c2 * (p_agents - p_dynamic)

    def laplacian(self, q_agents):
        L = torch.zeros(self.na, self.na)
        for i in range(self.na):
            r_sigma = self.sigma_norm(self.r, self.e)
            for j in range(self.na):
                if i != j:
                    z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2], self.e)
                    L[i, j] = self.rho_function(z_sigma / r_sigma, self.ha)
        return torch.diag(torch.sum(L, 1)) - L

    def augmented_laplacian(self, q_agents):
        L = self.laplacian(q_agents)
        self.save_laplacians.append(L)
        return torch.kron(L, torch.eye(2))

    def grad_V(self, L, q_agents):
        grad_V  = torch.zeros(2 * self.na)
        for i in range(self.na):
            r_sigma = self.sigma_norm(self.r, self.e)
            d_sigma = self.sigma_norm(self.d, self.e)
            for j in range(self.na):
                if i != j and L[2 * i, 2 * j] != 0:
                    z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2], self.e)
                    n_ij    = (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]) / (torch.sqrt(1 + self.e * (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2) ** 2))
                    grad_V[2 * i:2 * i + 2] -= self.phi_alpha_function(z_sigma, r_sigma, self.ha, d_sigma, self.a, self.b, self.c) * n_ij
        return grad_V

    def flocking_dynamics(self, t, inputs):
        L = self.augmented_laplacian(inputs[:2 * self.na])
        DeltaV = self.grad_V(L, inputs[:2 * self.na])

        dq = inputs[2 * self.na:4 * self.na]
        dp = -DeltaV - L @ inputs[2 * self.na:4 * self.na] + self.f_control(inputs[:2 * self.na], inputs[2 * self.na:4 * self.na], inputs[4 * self.na:6 * self.na], inputs[6 * self.na:])

        return dq, dp

    def leader_dynamics(self, t, inputs):
        if self.TV:
            velocity = torch.zeros(2 * self.na)
            acceleration = torch.zeros(2 * self.na)
            velocity[::2] = self.vx
            velocity[1::2] = -self.A*self.w*torch.sin(self.w*t + self.phi)
            acceleration[::2] = torch.zeros(self.na)
            acceleration[1::2] = -self.A*self.w*self.w*torch.cos(self.w*t + self.phi)
            return velocity, acceleration
        else:
            return inputs[6 * self.na:], torch.zeros(2 * self.na)

    def overall_dynamics(self, t, inputs):
        da = self.flocking_dynamics(t, inputs)
        dd = self.leader_dynamics(t, inputs)
        print(f'simulation time is {t.item()}')
        return torch.cat((da[0],
                          da[1] + torch.normal(mean=torch.zeros(da[1].shape[0]), std=self.noise*torch.ones(da[1].shape[0])),
                          dd[0],
                          dd[1]))

    def sample(self, inputs, simulation_time, step_size):
        targets = odeint(self.overall_dynamics, inputs, simulation_time, method='euler', options={'step_size': step_size})
        return targets


def generate_agents(na):
    return 10.0 * torch.rand(2 * na) - 5.0, torch.zeros(2 * na)


def generate_agents_test(na):
    return 0.5 * torch.randn(2 * na), torch.zeros(2 * na)


def generate_leader(na):
    return torch.zeros(2 * na), torch.zeros(2 * na)


