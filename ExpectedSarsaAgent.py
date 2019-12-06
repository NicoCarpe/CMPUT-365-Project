import numpy as np
from tiles3 import tiles, IHT
from utils import argmax
from agent import BaseAgent


class MountainCarTileCoder:
    def __init__(self, iht_size, num_tilings, num_tiles):
        '''
        Initializes the MountainCar Tile Coder
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tillings
        num_tiles -- int, the number of tiles. Both width and height of the tile coder are the same.

        Class Variables:
        self.iht -- IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        '''

        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, position, velocity):
        '''
        Takes a position and velocity from the MountainCar Environment and returns a numpy array of active tiles.

        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.007 and 0.007

        returns -- np.array, active tiles
        '''
        # Set the max and min of position and velocity to scale the input
        min_position = -1.2
        max_position = 0.5
        min_velocity = -0.07
        max_velocity = 0.07

        # Scale position and velocity
        scale_position = self.num_tiles / (max_position - min_position)
        scale_velocity = self.num_tiles / (max_velocity - min_velocity)

        active_tiles = tiles(self.iht, self.num_tilings, [scale_position * position, scale_velocity * velocity])
        return np.array(active_tiles)


class ExpectedSarsaAgent(BaseAgent):
    
    def __init__(self):
        # Initialization of Sarsa Agent. All values are set to None so they can be initialized in the agent_init method.
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.iht_size = None
        self.w = None
        self.alpha = None
        self.num_tilings = None
        self.num_tiles = None
        self.mctc = None
        self.initial_weights = None
        self.num_actions = None
        self.previous_tiles = None

    def agent_init(self, agent_info={}):
        # Setup for the agent called when the experiment first starts.
        self.num_tilings = agent_info.get("num_tilings")
        self.num_tiles = agent_info.get("num_tiles")
        self.iht_size = agent_info.get("iht_size")
        self.epsilon = agent_info.get("epsilon")
        self.gamma = agent_info.get("gamma")
        self.alpha = agent_info.get("alpha")
        self.initial_weights = agent_info.get("initial_weights")
        self.num_actions = agent_info.get("num_actions")

        # Initialize self.w to num_actions times the iht_size. Because we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        # Initialize self.mctc to MountainCarTileCoder
        self.mctc = MountainCarTileCoder(self.iht_size, self.num_tilings, self.num_tiles)

    def select_action(self, tiles):
        # Selects an action using epsilon greedy
        # set action_values to a empty list
        # chosen_action to None
        action_values = []
        chosen_action = None

        # Loop through the weights to get action value for each action
        for action in range(self.num_actions):
            value = 0
            for active_tile in tiles:
                value += self.w[action][active_tile]
            # append action value of each action into action_values list
            action_values.append(value)

        # Use epsilon greedy to select an action     
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)
        else:
            chosen_action = argmax(action_values)
        # return chosen_action and the action value of the chosen action
        return chosen_action, action_values[chosen_action], action_values

    def agent_start(self, state):
        # set position and velocity from state
        position, velocity = state
        
        # use self.mctc to set active_tiles using position and velocity 
        active_tiles = self.mctc.get_tiles(position, velocity)

        # set current_action, action_value and action_values(list of action value if all actions) using select_action function with active_tiles
        current_action, action_value, action_values = self.select_action(active_tiles)

        # set self.last_action_value and self.last_action_values
        self.last_action_value = action_value
        self.last_action_values = action_values

        # set self.last_action and self.previous_tiles
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        # return self.last_action
        return self.last_action

    def agent_step(self, reward, state):
        # set position and velocity from state
        position, velocity = state

        # use self.mctc to set active_tiles using position and velocity 
        active_tiles = self.mctc.get_tiles(position, velocity)
        # set current_action, action_value and action_values(list of action value if all actions) using select_action function with active_tiles
        current_action, action_value, action_values = self.select_action(active_tiles)
        # initalized feature_vector to all zeors
        feature_vector = np.zeros(self.iht_size)
        # loop through self.previous_tiles
        for i in self.previous_tiles:
            # set the value of index = to active tile in feature vecotor to 1 else 0.
            feature_vector[i] = 1
        # initalized target to 0
        target = 0
        # loop through all the actions
        # using epsilon greedy policy to get policy
        for action in range(self.num_actions):
            if (action == np.argmax(action_values)):
                policy = 1 - self.epsilon + (self.epsilon / self.num_actions)
            else:
                policy = self.epsilon / self.num_actions
            # sum the target of all actions
            # target = policy of next state, next action times value of next state next action and weights
            target += policy * action_values[action]
        # updated self.w of currrent action
        self.w[self.last_action] = self.w[self.last_action] + self.alpha * (reward + self.gamma * target - self.last_action_value) * feature_vector
        # set self.last_action
        self.last_action_value = action_value

        # set self.last_action and self.previous_tiles
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self, reward):
        # initalized feature_vector to all zeors
        feature_vector = np.zeros(self.iht_size)
        for i in self.previous_tiles:
            feature_vector[i] = 1
        # updated self.w (No next state and next action)
        self.w[self.last_action] = self.w[self.last_action] + self.alpha * (reward - self.last_action_value) * feature_vector

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass
