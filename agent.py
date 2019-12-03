import numpy as np
from tiles3 import tiles, IHT
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
        self.num_tillings -- int, the number of tilings the tile coder will use
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
