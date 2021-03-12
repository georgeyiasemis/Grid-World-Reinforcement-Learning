import numpy as np
import matplotlib.pyplot as plt
import random
from environment import *

def policy_evaluation(grid, policy, gamma=1.0, threshold=0.0001):
    # Make sure policy has right dimensions
    assert policy.shape == (grid.state_size, grid.action_size)
    # Make sure delta is bigger than the threshold to start with
    delta = 3 * threshold

    # Get the reward and transition matrices
    R = grid.get_reward_matrix()
    T = grid.get_transition_matrix()

    # The value is initialised at 0
    V = np.zeros(grid.state_size)

    V_new = V.copy()

    epochs = 0

    # While the Value has not yet converged do:
    while delta > threshold:
        epochs += 1
        for state_idx in range(grid.state_size):
            # If it is one of the absorbing states, ignore
            if grid.absorbing[0, state_idx]:
                continue

            # Accumulator variable for the Value of a state
            tmpV = 0
            for action_idx in range(grid.action_size):
                # Accumulator variable for the State-Action Value
                tmpQ = 0
                for state_idx_prime in range(grid.state_size):
                    tmpQ += T[state_idx_prime, state_idx, action_idx] * \
                            (R[state_idx_prime, state_idx, action_idx] + gamma * V[state_idx_prime])

                tmpV += policy[state_idx, action_idx] * tmpQ

            # Update the value of the state
            V_new[state_idx] = tmpV

        # After updating the values of all states, update the delta
        delta =  np.abs(V_new - V).max()
        # and save the new value into the old
        V = V_new.copy()

    return V, epochs

def policy_iteration(grid, gamma=1.0,threshold=0.0001):

    policy = np.zeros((grid.state_size, grid.action_size))
    # Initialise a random policy
    policy[:, 0] = 1.0

    T = grid.get_transition_matrix()
    R = grid.get_reward_matrix()

    epochs = 0
    while True:

        # Policy Evaluation
        V, epochs_eval = policy_evaluation(grid, policy, gamma, threshold)

        epochs += epochs_eval

        # Policy Improvement
        policy_stable = True

        for state_idx in range(grid.state_size):

            if grid.absorbing[0, state_idx]:
                continue

            old_action = np.argmax(policy[state_idx, :])

            Q = np.zeros(grid.action_size)

            for state_idx_prime in range(grid.state_size):
                Q += T[state_idx_prime, state_idx, :] * (R[state_idx_prime, state_idx, :] + gamma * V[state_idx_prime])

            new_policy = np.zeros(grid.action_size)
            new_policy[np.argmax(Q)] = 1.0
            policy[state_idx] = new_policy

            if old_action != np.argmax(policy[state_idx]):
                policy_stable = False

        if policy_stable:
            return V, policy, epochs

def value_iteration(grid, gamma=1.0, threshold=0.0001):

    # The value is initialised at 0
    V = np.zeros(grid.state_size)

    # Get the reward and transition matrices
    T = grid.get_transition_matrix()
    R = grid.get_reward_matrix()

    epochs = 0
    while True:

        epochs += 1
        delta = 0

        for state_idx in range(grid.state_size):
            if grid.absorbing[0, state_idx]:
                continue

            v = V[state_idx]

            Q = np.zeros(grid.action_size)
            for state_idx_prime in range(grid.state_size):

                Q += T[state_idx_prime, state_idx, :] * \
                    (R[state_idx_prime, state_idx, :] + gamma * V[state_idx_prime])

            V[state_idx]= np.max(Q)
            delta = max(delta, np.abs(v - V[state_idx]))

        if delta < threshold:

            optimal_policy = np.zeros((grid.state_size, grid.action_size))
            for state_idx in range(grid.state_size):
                Q = np.zeros(grid.action_size)
                for state_idx_prime in range(grid.state_size):
                    Q += T[state_idx_prime, state_idx, :] * \
                        (R[state_idx_prime, state_idx, :] + gamma * V[state_idx_prime])

                optimal_policy[state_idx, np.argmax(Q)] = 1


            return optimal_policy, epochs

def MC_policy_evaluation(grid, policy, num_episodes, gamma=1.0):

    V = [0.0 for i in range(grid.state_size)]
    R = [[] for i in range(grid.state_size)]

    for i in range(num_episodes):
        starting_loc = random.choice(list(set(grid.locs)-set(grid.absorbing_locs)))
        starting_state = grid.loc_to_state(starting_loc, grid.locs)

        trace = grid.sample_episode(policy, starting_loc, gamma=gamma, max_episode_len=100)

        G = 0
        state_trace = ([s[0] for s in trace])
        visited_states = set(transition[0] for transition in trace) # first visit MC

        for state in visited_states:
            state_1st_occurance_idx = state_trace.index(state)

            G = [transition[2] * gamma ** k for (k, transition) in enumerate(trace[state_1st_occurance_idx:])]
            G = np.sum(G)
            R[grid.loc_to_state(state, grid.locs)].append(G)
            V[grid.loc_to_state(state, grid.locs)] = np.mean(R[grid.loc_to_state(state, grid.locs)])

    return np.array(V)


def TD_estimation(grid, policy, num_episodes, gamma=1.0):

    V = [0.0 for i in range(grid.state_size)]
    # State visit counter
    C = [0 for i in range(grid.state_size)]

    for i in range(num_episodes):
        starting_loc = random.choice(list(set(grid.locs)-set(grid.absorbing_locs)))
        starting_state = grid.loc_to_state(starting_loc, grid.locs)

        trace = grid.sample_episode(policy, starting_loc, gamma=gamma, max_episode_len=100)
        for transition in trace:
            state = grid.loc_to_state(transition[0], grid.locs)
            C[state] += 1
            state_prime = grid.loc_to_state(transition[-1], grid.locs)
            reward = transition[2]
            td_target = reward + gamma * V[state_prime]
            td_error = td_target - V[state]
            alpha_t = 1 / C[state]
            # Gradually forget older visits
            V[state] += alpha_t * td_error

    return np.array(V)
