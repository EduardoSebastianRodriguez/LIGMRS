import torch
from flockingFunctions import realSystem, generate_leader, generate_agents
import matplotlib.pyplot as plt

TV = False

numSamples = 150
seed = 2
numAgents = 50

# Set seed
torch.manual_seed(seed)

# Hyperparameters
step_size = 0.04
time = numSamples * step_size
simulation_time = torch.linspace(0, time-step_size, int(time/step_size))

# Parameters
na = torch.as_tensor(numAgents)  # Number of agents
d = torch.as_tensor(0.7)  # Desired flocking distance
r = torch.as_tensor(1.2)  # Radius of influence
e = torch.as_tensor(0.1)  # For the sigma-norm
a = torch.as_tensor(5.0)  # For the sigmoid
b = torch.as_tensor(5.0)  # For the sigmoid
c = torch.abs(a - b) / torch.sqrt(4 * a * b)  # For the sigmoid
ha = torch.as_tensor(0.2)  # For the sigmoid
c1 = torch.as_tensor(0.4)  # Tracking gain 1
c2 = torch.as_tensor(0.8)  # Tracking gain 2
noise = torch.as_tensor(0.005) # Agent dynamics are affected by a ZMG noise with std equal to "noise"

parameters = {"na": na,
              "d": d,
              "r": r,
              "e": e,
              "a": a,
              "b": b,
              "c": c,
              "ha": ha,
              "c1": c1,
              "c2": c2,
              "noise": noise,
              "TV": TV}

# Initialize the system to learn
real_system = realSystem(parameters)

# Build training dataset
q_agents, p_agents = generate_agents(na)
q_dynamic, p_dynamic = generate_leader(na)
input = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
trajectory = real_system.sample(input, simulation_time, step_size)
trajectory = trajectory[:, :4*numAgents]

# Store data
torch.save(trajectory, 'flocking_trajectories'+str(numAgents)+str(numSamples)+str(seed)+str(d)+str(r)+'_.pth')
torch.save(real_system.save_laplacians, 'flocking_laplacians'+str(numAgents)+str(numSamples)+str(seed)+str(d)+str(r)+'_.pth')

plt.figure()
for i in range(numAgents):
    plt.plot(trajectory[:, 2*i], trajectory[:, 2*i+1], 'b')
    plt.plot(trajectory[0, 2 * i], trajectory[0, 2 * i + 1], 'gs')
    plt.plot(trajectory[-1, 2 * i], trajectory[-1, 2 * i + 1], 'ro')
plt.show()
