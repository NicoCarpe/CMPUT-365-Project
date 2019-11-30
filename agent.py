import numpy as np
from tiles3 import tiles, IHT
from agent import BaseAgent


class semi_gradient_ExpectedSarsa(BaseAgent):
    def __init__(self):
        self.min_position = None
        self.max_position = None
        self.min_velocity = None
        self.max_velocity = None
        self.numTilings = None
        self.step_size = None
        self.discount_factor = None
        self.memorySize = None
        self.epsilon = None

    def agent_init(self, agent_info={}):
        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # set class attributes
        # self.num_states = agent_info.get("num_states")
        # self.num_groups = agent_info.get("num_groups")
        self.min_position = agent_info("min_position")
        self.max_position = agent_info("max_position")
        self.min_velocity = agent_info("min_velocity")
        self.max_velocity = agent_info("max_velocity")

        self.step_size = agent_info.get("step_size")
        self.discount_factor = agent_info.get("discount_factor")
        self.memorySize = agent_info.get("memorySize")
        self.numTilings = agent_info.get("numTilings")
        self.epsilon = agent_info.get("epsilon")

        self.size_of_tiling = self.numTilings * self.numTilings
        self.alpha = 0.1 / self.numTilings

        # self.all_state_fetures = np.array([])

        # initialize weights
        self.weights = np.zeros(self.size_of_tiling)
        for i in range(len(self.weights)):
            self.weights[i] = np.random.uniform(-0.001, 0)

    def get_fetures(self, position, velocity):
        position_scale = self.numTilings / (self.max_position - self.min_position)
        velocity_scale = self.numTilings / (self.max_velocity - self.min_velocity)
        iht = IHT(self.memorySize)
        return tiles(iht, self.numTilings, [velocity_scale * velocity, position_scale * position_scale])
        
