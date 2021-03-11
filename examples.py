import numpy as np

from environment import GridWorld
from methods import *

# Examples of Policy Evaluation, Policy Iteration

# Define Environment
grid = GridWorld()


#####################################
######## Policy Evaluation ##########
#####################################

# Define Uniform policy
policy = np.ones((grid.state_size, grid.action_size)) * 0.25

# Discount factor and Error threshold
gamma = 0.9
threshold = 0.00001

State_Value_fun, epochs = policy_evaluation(grid, policy, gamma, threshold)

print("Policy Evaluation \n\nPolicy Value:")
for state in range(grid.state_size):
    print('State s{}: Value = {}'.format(str(state), State_Value_fun[state]))

print("\nTotal Epochs = {}\n".format(epochs))

# Plot gamma against computation time
gammas = np.linspace(0, 1, 11)
times = []
for gamma in gammas:
    V, epochs = policy_evaluation(grid, policy, gamma, threshold)
    times.append(epochs)

plt.figure()
plt.plot(gammas, times)
plt.xlabel('Epochs', fontsize=15); plt.ylabel(r'$\gamma$', fontsize=15)
plt.show()
# plt.savefig('./gammas_vs_epochs.png')

#####################################
######## Policy Iteration ###########
#####################################

# Discount factor and Error threshold
gamma = 0.9
threshold = 0.00001

State_val_fun_opt, policy_opt, epochs = policy_iteration(grid, gamma, threshold)

print("Policy Iteration \n")
print("Optimal Policy:\n", policy_opt,'\n')

for state in range(grid.state_size):
    print('State s{}: Value = {}'.format(str(state), State_Value_fun[state]))

print("\nTotal Epochs = {}".format(epochs))
