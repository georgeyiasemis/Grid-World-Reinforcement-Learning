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


def TD_estimation(grid, policy, num_episodes, gamma=1.0, alpha=None):
    # Init state value function
    V = [0.0 for i in range(grid.state_size)]
    # State visit counter
    C = [0 for i in range(grid.state_size)]

    for i in range(num_episodes):
        # Start at random location
        starting_loc = random.choice(list(set(grid.locs)-set(grid.absorbing_locs)))
        # Sample a trace
        trace = grid.sample_episode(policy, starting_loc, gamma=gamma, max_episode_len=100)
        for transition in trace:
            state = grid.loc_to_state(transition[0], grid.locs)
            C[state] += 1
            state_prime = grid.loc_to_state(transition[-1], grid.locs)
            reward = transition[2]
            td_target = reward + gamma * V[state_prime]
            td_error = td_target - V[state]

            alpha_t = 1 / C[state] if alpha == None else alpha
            # Gradually forget older visits
            V[state] += alpha_t * td_error

    return np.array(V)

def on_policy_eps_greedy_MC_control(grid, policy, num_episodes, gamma=1.0, epsilon=1.0):
    # Init state-action function to zeros
    Q = np.zeros((grid.state_size, grid.action_size))
    # Returns
    R = [[0.0 for j in range(grid.action_size)] for i in range(grid.state_size)]
    # Keep counter to calculate the average return
    C = [[0 for j in range(grid.action_size)] for i in range(grid.state_size)]

    for i in range(num_episodes):
        # Start at random location
        starting_loc = random.choice(list(set(grid.locs) - set(grid.absorbing_locs)))
        # Sample a trace
        trace = grid.sample_episode(policy, starting_loc, gamma=gamma, max_episode_len=100)
        # Encourage exploration at first episodes
        epsilon = max(epsilon * 0.99995, 0.05)
        # Create a state, action trace
        trace_state_actions = [(s, a) for (s, a, _, _) in trace]
        # List of visited states
        visited_states = list(set(grid.loc_to_state(transition[0], grid.locs) for transition in trace))
        visited_states.sort()
        # List of visited state-action pairs
        visited_state_actions = [state_action for t, state_action in \
                            enumerate(trace_state_actions) if state_action not in trace_state_actions[:t]]

        # Since we are doing first visit MC
        for state_action in visited_state_actions:
            state_action_1st_occurance_idx = trace_state_actions.index(state_action)
            G = np.sum([transition[2] * gamma ** k for (k, transition) in \
                    enumerate(trace[state_action_1st_occurance_idx:])])
            state_loc, action = state_action
            state = grid.loc_to_state(state_loc, grid.locs)
            R[state][grid.action_to_idx(action)] += G
            C[state][grid.action_to_idx(action)] += 1
            Q[state, grid.action_to_idx(action)] = R[state][grid.action_to_idx(action)] / C[state][grid.action_to_idx(action)]

        # Make an epsilon-greedy policy
        best_actions = np.argmax(Q[visited_states], 1)
        policy[visited_states, best_actions] = 1 - epsilon + epsilon / grid.action_size
        other_actions = [[i for i in range(grid.action_size) if i!= j] for j in best_actions]
        other_actions = np.array(other_actions).T
        policy[visited_states, other_actions] = epsilon / grid.action_size

    return Q, policy

def MC_iterative_eps_greedy_control(grid, num_episodes, gamma=1.0, epsilon=0.9, alpha=0.1):
    # Init state-action function to zeros
    Q = np.zeros((grid.state_size, grid.action_size))
    # Init eps-greedy policy
    policy = np.ones((grid.state_size, grid.action_size)) * 0.25

    for i in range(num_episodes):
        # Start at random location
        starting_loc = random.choice(list(set(grid.locs) - set(grid.absorbing_locs)))
        # Sample a trace
        trace = grid.sample_episode(policy, starting_loc, gamma=gamma, max_episode_len=100)
        # Encourage exploration at first episodes
        epsilon = max(epsilon * 0.99995, 0.05)
        # Create a state, action trace
        trace_state_actions = [(s, a) for (s, a, _, _) in trace]
        # List of visited states
        visited_states = list(set(grid.loc_to_state(transition[0], grid.locs) for transition in trace))
        visited_states.sort()
        # List of visited state-action pairs
        visited_state_actions = [state_action for t, state_action in \
                            enumerate(trace_state_actions) if state_action not in trace_state_actions[:t]]

        # Since we are doing first visit MC
        for state_action in visited_state_actions:
            state_action_1st_occurance_idx = trace_state_actions.index(state_action)
            G = np.sum([transition[2] * gamma ** k for (k, transition) in \
                    enumerate(trace[state_action_1st_occurance_idx:])])
            state_loc, action = state_action
            state = grid.loc_to_state(state_loc, grid.locs)
            Q[state, grid.action_to_idx(action)] += alpha * (G - Q[state, grid.action_to_idx(action)])

        # Make an epsilon-greedy policy
        best_actions = np.argmax(Q[visited_states], 1)
        policy[visited_states, best_actions] = 1 - epsilon + epsilon / grid.action_size
        other_actions = [[i for i in range(grid.action_size) if i!= j] for j in best_actions]
        other_actions = np.array(other_actions).T
        policy[visited_states, other_actions] = epsilon / grid.action_size

    return Q, policy

def SARSA(grid, num_episodes,  max_episode_len=100, gamma=1.0, epsilon=1.0, alpha=0.1):

    # Init state-action function to zeros
    Q = np.zeros((grid.state_size, grid.action_size))

    for i in range(num_episodes):
        # Start at random location
        state_loc = random.choice(list(set(grid.locs) - set(grid.absorbing_locs)))
        state_idx = grid.loc_to_state(state_loc, grid.locs)

        # Choose a' from s' using policy derived from Q (epsilon-greedy)
        action_idx = epsilon_greedy_action(Q, epsilon, state_idx)

        steps = 0
        while (steps <= max_episode_len) & (not grid.absorbing[0, state_idx]):

            steps += 1

            # Take a step : take action a', observe reward and s'
            _, _, reward, state_prime_idx = grid.state_action_step(state_idx, action_idx)

            # Choose a' from s' using policy derived from Q (epsilon-greedy)
            action_prime_idx = epsilon_greedy_action(Q, epsilon, state_prime_idx)

            # Update Q function
            Q[state_idx, action_idx] += alpha * \
                    (reward + gamma * Q[state_prime_idx, action_prime_idx] - Q[state_idx, action_idx])

            state_idx, action_idx = state_prime_idx, action_prime_idx

        # Encourage exploration at first episodes
        epsilon = max(epsilon * 0.99995, 0.05)

    policy = Q.argmax(1)

    return Q, policy

def Q_learning(grid, num_episodes,  max_episode_len=100, gamma=1.0, epsilon=1.0, alpha=0.1):

    # Init state-action function to zeros
    Q = np.zeros((grid.state_size, grid.action_size))

    for i in range(num_episodes):
        # Start at random location
        state_loc = random.choice(list(set(grid.locs) - set(grid.absorbing_locs)))
        state_idx = grid.loc_to_state(state_loc, grid.locs)

        steps = 0
        while (steps <= max_episode_len) & (not grid.absorbing[0, state_idx]):

            steps += 1

            # Choose a' from s' using policy derived from Q (epsilon-greedy)
            action_idx = epsilon_greedy_action(Q, epsilon, state_idx)

            # Take a step : take action a', observe reward and s'
            _, _, reward, state_prime_idx = grid.state_action_step(state_idx, action_idx)

            # Update Q function (choose greedy action from s')
            Q[state_idx, action_idx] += alpha * \
                    (reward + gamma * Q[state_prime_idx].max() - Q[state_idx, action_idx])

            state_idx = state_prime_idx

        # Encourage exploration at first episodes
        epsilon = max(epsilon * 0.99995, 0.05)

    policy = Q.argmax(1)

    return Q, policy

def epsilon_greedy_policy(Q, epsilon):
    policy = np.zeros(Q.shape)
    best_actions = np.argmax(Q, 1)
    policy[range(Q.shape[0]), best_actions] = 1 - epsilon + epsilon / Q.shape[1]
    other_actions = [[i for i in range(Q.shape[1]) if i!= j] for j in best_actions]
    other_actions = np.array(other_actions).T
    policy[range(Q.shape[0]), other_actions] = epsilon / Q.shape[1]
    return policy

def epsilon_greedy_action(Q, epsilon, state_idx):

    if np.random.rand() < 1 - epsilon:
        # p(a = a*|s) = 1 - epsilon + epsilon / |A(s)|
        action_idx = Q[state_idx].argmax()
    else:
        # p(a = a', a'!= a*|s) = epsilon / |A(s)|
        action_idx = np.random.choice(range(Q.shape[1]))
    return action_idx
