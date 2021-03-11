import numpy as np

from environment import GridWorld
from methods import MC_policy_evaluation

# Run First Visit Monte Carlo Policy Evaluation for a uniform policy
# We suppose that the transition matrix is unknown (model-free)

# Define Environment
grid = GridWorld()

# Define Uniform policy
policy = np.ones((grid.state_size, grid.action_size)) * 0.25

# Number of episodes
num_episodes = 2000

# Discount factor
gamma = 0.9

State_Value_fun = MC_policy_evaluation(grid, policy, num_episodes, gamma)

print("Policy Value:")
for state in range(grid.state_size):
    print('State s{}: Value = {}'.format(str(state), State_Value_fun[state]))
