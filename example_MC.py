import numpy as np

from environment import GridWorld
from methods import MC_policy_evaluation, on_policy_eps_greedy_MC_control

# We suppose that the transition matrix is unknown (model-free)

# Run First Visit Monte Carlo Policy Evaluation for a uniform policy
print('FV MC Policy Evaluation\n')

# Define Environment
grid = GridWorld()

# Define Uniform policy
policy = np.ones((grid.state_size, grid.action_size)) * 0.25

# Number of episodes
num_episodes = 1000

# Discount factor
gamma = 0.9

State_Value_fun = MC_policy_evaluation(grid, policy, num_episodes, gamma)

print("Policy Value:")
for state in range(grid.state_size):
    print('State s{}: Value = {}'.format(str(state), State_Value_fun[state]))

# On policy ε-greedy First Visit MC control
print('\nFV MC e-greedy control\n')
# Epsilon
epsilon = 0.7

# Learning Rate
alpha = 0.5

Q, policy = on_policy_eps_greedy_MC_control(grid, policy, num_episodes, gamma, epsilon)

policy = np.argmax(policy, 1)
# Draw policy
grid.draw_deterministic_policy(policy)
