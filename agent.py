import numpy as np
from tiles3 import tiles, IHT
from agent import BaseAgent

def get_fetures(maxSize, numTilings, position, velocity):
    iht = IHT(maxSize)
    return tiles(iht, numTilings, [velocity, position])

class semi_gradient_ExpectedSarsa(BaseAgent):
    def __init__(self):
        self.num_states = None
        self.num_groups = None
        self.step_size = None
        self.discount_factor = None

    def agent_init(self, agent_info={}):
        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # set class attributes
        self.num_states = agent_info.get("num_states")
        # self.num_groups = agent_info.get("num_groups")
        self.step_size = agent_info.get("step_size")
        self.discount_factor = agent_info.get("discount_factor")
        self.memorySize = agent_info.get("memorySize")
        self.numTilings = agent_info.get("numTilings")
        self.maxSize = agent_info.get("maxsize")

        self.tile_step_size = self.step_size / self.numTilings

        # self.all_state_fetures = np.array([])

        # initialize weights
        self.weights = np.zeros(self.maxSize)