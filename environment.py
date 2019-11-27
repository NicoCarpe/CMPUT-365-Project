import math
import numpy as np
from environment import BaseEnvironment


class MountainEnvironment(BaseEnvironment):

    def env_init(self, env_info={}):
        '''
        Assume env_info dict conatins:
        {
            min_posiiton: -1.2[int],
            max_position: 0.5[int],
            min_velocity: -0.07[int],
            max_velosity: 0.07[int],
            gravity: 0.0025[int],
            start_position: np.random.sample[-0.6, -0.4] [int],
            start_velosity: 0[int],
            action_dicount: 0.001[int],
            seed: [int]
        }
        '''
        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed"))
        
        # set each class attribute
        self.min_position = env_info["min_position"]
        self.max_position = env_info["max_position"]
        self.min_velocity = env_info["min_velocity"]
        self.max_velocity = env_info["max_velocity"]
        self.gravity = env_info["gravity"]
        self.start_position = env_info["start_position"]
        self.start_velocity = env_info["start_velocity"]
        self.action_dicount = env_info["action_dicount"]
         
    def env_start(self):
        # set starting reward, position, and velocity
        reward = 0.0
        position = self.start_position
        velocity = self.start_velocity
        # set position and velocity to a tuple called state
        state = (position, velocity)
        # set is_terminal to False
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        return self.reward_state_term[1]

    def env_step(self, action):
        # set last_postion and last_velocity from self.reward_state_term
        last_state = self.reward_state_term[1]
        last_position = last_state[0]
        last_velocity = last_state[1]

        # updated velocity
        velocity = last_velocity + self.action_dicount * action - self.gravity * math.cos(3 * last_position)
        # checked if velocity is whithin envionment bounds
        if velocity > self.max_velocity:
            velocity = self.max_velocity
        if velocity < self.min_velocity:
            velocity = self.min_velocity
        
        # updated position
        position = last_position + velocity
        # checked position reached goal
        if position >= self.max_position:
            reward = -1
            is_terminal = True
        # checked position within environment bounds 
        elif position < self.min_position:
            # checked if velocity needs to be reset
            if velocity < 0:
                velocity = 0
                position = self.min_position
            else:
                position = self.min_position
            reward = -1
            is_terminal = False
        # position in bounds
        else:
            reward = -1
            is_terminal = False
        
        # updated self.reward_state_term
        state = (position, velocity)
        self.reward_state_term = (reward, state, is_terminal)

        return self.reward_state_term
            

        

