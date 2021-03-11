import numpy as np

from environment import GridWorld
from methods import *

#####################################
########## Value Iteration ##########
#####################################

# Define Environment
grid = GridWorld()

# Discount factor and Error threshold
gamma = 0.9
threshold = 0.00001

optimal_policy, epochs = value_iteration(grid, gamma, threshold)

print("Policy Iteration \n")
print("Optimal Policy:\n", optimal_policy)
print("\nTotal Epochs = {}".format(epochs))

# Draw Optimal Policy
policy = np.array([np.argmax(optimal_policy[i, :]) for i in range(grid.state_size)])
grid.draw_deterministic_policy(policy)
plt.title('Optimal Policy for Value Iteration')
plt.savefig('./opt_policy_VI.png')
