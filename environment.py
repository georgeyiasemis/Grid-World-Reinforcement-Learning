import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):

    def __init__(self, paint_maps=False):

        '''
        Code adapted from Reinforcement Learning Course at
        Imperial College London, Autumn 2019.
        '''

        ### Attributes defining the Gridworld #######
        # Shape of the gridworld
        self.shape = (5, 5)

        # Locations of the obstacles
        self.obstacle_locs = [(1, 1), (2, 1), (2, 3)]

        # Locations for the absorbing states
        self.absorbing_locs = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

        # Rewards for each of the absorbing states
        self.special_rewards = [10, -100, -100, -100, 10] #corresponds to each of the absorbing_locs

        # Reward for all the other states
        self.default_reward = -5

        # Starting location, can be reset using the corresponding method
        self.starting_loc = (3, 0)

        # Action names
        self.action_names = ['N','E','S','W']

        # Number of actions
        self.action_size = len(self.action_names)


        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = [0.4, 0.2, 0.2 , 0.2]

        ######################### Internal State  ##############################

        # Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()

        # Number of valid states in the gridworld (not walls)
        self.state_size = state_size

        # Transition operator (3D tensor)
        self.T = T

        # Reward function (3D tensor)
        self.R = R

        # Absorbing states
        self.absorbing = absorbing

        # The locations of the valid states
        self.locs = locs

        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs);

        # Locating the initial state
        self.initial = np.zeros((1, len(locs)));
        self.initial[0, self.starting_state] = 1


        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob] = 1

        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1

        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]

        #Illustrating the grid world
        if paint_maps:
            self.paint_maps()

    def reset_starting_loc(self, loc, paint=False):
        assert loc not in self.obstacle_locs and loc not in self.absorbing_locs
        self.starting_loc = loc
        self.starting_state = self.loc_to_state(self.starting_loc, self.locs);

        # Locating the initial state
        self.initial = np.zeros((1, len(self.locs)));
        self.initial[0, self.starting_state] = 1
        if paint == True:
            self.paint_maps()


    ################################ Getters ###################################
    def get_transition_matrix(self):
        return self.T

    def get_reward_matrix(self):
        return self.R
    ############################################################################

    def draw_deterministic_policy(self, Policy):
        # Draw a deterministic policy
        # The policy needs to be a np array of self.state_size values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure()
        plt.imshow(self.walls + self.rewarders + self.absorbers)

        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
        for state, action in enumerate(Policy):
            if self.absorbing[0, state]:
                continue
            action_arrow = arrows[action]
            location = self.locs[state]
            plt.text(location[1], location[0], action_arrow, ha='center', va='center')

        plt.show()
        plt.savefig('./det_policy.png')

    ############################################################################



    ###################### Internal Helper Functions ###########################
    def paint_maps(self, savefig=True):
        plt.figure(figsize=(16,8))

        plt.subplot(1,4,1)
        plt.imshow(self.walls)
        plt.title('Walls')

        plt.subplot(1,4,2)
        plt.imshow(self.absorbers)
        plt.title('Absorbing States')

        plt.subplot(1,4,3)
        plt.imshow(self.rewarders)
        plt.title('Rewarding States')

        plt.subplot(1,4,4)
        plt.imshow(self.rewarders + self.absorbers)
        plt.title('States')
        for state in range(self.state_size):
            location = self.locs[state]
            plt.text(location[1], location[0], 's{}'.format(state), ha='center', va='center')

        plt.show()

        if savefig:
            plt.savefig('./gridworld.png')

    def build_grid_world(self):
        # Get the locations of all the valid states, the neighbours of each state (by state number),
        # and the absorbing states (array of 0's with ones in the absorbing states)
        locations, neighbours, absorbing = self.get_topology()

        # Get the number of states
        S = len(locations)

        # Initialise the transition matrix
        T = np.zeros((S, S, 4))

        for action in range(4):
            for effect in range(4):

                # Randomize the outcome of taking an action
                outcome = (action + effect + 1) % 4

                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1

                # Fill the transition matrix
                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state, outcome]
                    post_state = int(post_state)
                    T[post_state, prior_state, action] += prob


        # Build the reward matrix
        R = self.default_reward * np.ones((S, S, 4))

        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i], locations)
            R[post_state, :, :]= sr

        return S, T, R, absorbing, locations

    def get_topology(self):
        height = self.shape[0]
        width = self.shape[1]

        index = 1
        locs = []
        neighbour_locs = []
        directions = ['nr', 'ea', 'so', 'we']
        for i in range(height):
            for j in range(width):
                # Get the locaiton of each state
                loc = (i,j)

                #And append it to the valid state locations if it is a valid state (ie not absorbing)
                if self.is_location(loc):
                    locs.append(loc)

                    # Get an array with the neighbours of each state, in terms of locations
                    local_neighbours = [self.get_neighbour(loc, direction) for direction in directions]
                    neighbour_locs.append(local_neighbours)

        # translate neighbour lists from locations to states
        num_states = len(locs)
        state_neighbours = np.zeros((num_states, 4))

        for state in range(num_states):
            for direction in range(4):
                # Find neighbour location
                nloc = neighbour_locs[state][direction]

                # Turn location into a state number
                nstate = self.loc_to_state(nloc, locs)

                # Insert into neighbour matrix
                state_neighbours[state, direction] = nstate;


        # Translate absorbing locations into absorbing state indices
        absorbing = np.zeros((1, num_states))
        for a in self.absorbing_locs:
            absorbing_state = self.loc_to_state(a, locs)
            absorbing[0, absorbing_state] =1

        return locs, state_neighbours, absorbing

    def loc_to_state(self, loc, locs):
        # Takes list of locations and returns index corresponding to input loc
        return locs.index(tuple(loc))

    def state_to_loc(self, state):
        # Takes an index and returns the corresponding index
        return self.locs[state]

    def action_to_idx(self, action):
        actions_idx = {'nr': 0, 'ea': 1, 'so': 2, 'we': 3}
        return actions_idx[action]

    def is_location(self, loc):
        # It is a valid location if it is in grid and not obstacle
        if loc[0] < 0 or loc[1] < 0 or loc[0] > self.shape[0] - 1 or loc[1] > self.shape[1] - 1:
            return False
        elif loc in self.obstacle_locs:
            return False
        else:
             return True

    def get_neighbour(self, loc, direction):
        #Find the valid neighbours (ie that are in the grif and not obstacle)
        i = loc[0]
        j = loc[1]

        nr = (i-1,j)
        ea = (i,j+1)
        so = (i+1,j)
        we = (i,j-1)

        # If the neighbour is a valid location, accept it, otherwise, stay put
        if(direction == 'nr' and self.is_location(nr)):
            return nr
        elif(direction == 'ea' and self.is_location(ea)):
            return ea
        elif(direction == 'so' and self.is_location(so)):
            return so
        elif(direction == 'we' and self.is_location(we)):
            return we
        else:
            #default is to return to the same location
            return loc

    ############## Methods to move in the environment model-free ###############

    def step(self, loc, policy):
        # Make sure loc is not a wall
        assert loc in self.locs
        state_idx = self.loc_to_state(loc, self.locs)

        # Available actions/directions
        actions = ['nr', 'ea', 'so', 'we']

        # Take an action w.r.t. to the policy
        action = np.random.choice(actions, p=policy[state_idx])
        action_idx = actions.index(action)

        # Move to the next location/state
        loc_prime = self.get_neighbour(loc, action)
        state_prime_idx = self.loc_to_state(loc_prime, self.locs)

        # Observe reward
        reward = self.R[state_prime_idx, state_idx, action_idx]

        # Return transition (st, at, rt, st+1)
        transition = (state_idx, action, reward, state_prime_idx)

        return transition

    def state_action_step(self, state_idx, action_idx):
        # Make sure loc is not a wall
        assert self.state_to_loc(state_idx) in self.locs
        # state_idx = self.loc_to_state(loc, self.locs)

        # Available actions/directions
        actions = ['nr', 'ea', 'so', 'we']

        action = actions[action_idx]
        # Move to the next location/state
        loc_prime = self.get_neighbour(self.state_to_loc(state_idx), action)
        state_prime_idx = self.loc_to_state(loc_prime, self.locs)
        # Observe reward
        reward = self.R[state_prime_idx, state_idx, action_idx]
        # Return transition (st, at, rt, st+1)
        transition = (state_idx, action_idx, reward, state_prime_idx)

        return transition


    def sample_episode(self, policy, starting_loc, gamma, max_episode_len=30):
        # Samples an episode of maximum length max_episode_len

        # Set starting location
        self.reset_starting_loc(starting_loc)
        starting_state = self.loc_to_state(self.starting_loc, self.locs)

        # Init an empty list to store transitions
        episode = []

        state_loc = self.starting_loc

        episode_len = 0

        while episode_len <= max_episode_len:

            episode_len += 1

            # Take a step into the environment and store transition
            transition = self.step(state_loc, policy)
            state_loc = self.state_to_loc(transition[-1])

            episode.append([self.state_to_loc(transition[0]), transition[1],
                            transition[2], self.state_to_loc(transition[-1])])

            # If an absorbing state is reached the episode is completed
            if self.absorbing[0, transition[-1]]:
                return episode

        # If max_episode_len is reached end episode even without reaching an
        # absorbing state
        return episode

################################################################################
