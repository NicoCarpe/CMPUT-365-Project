
import numpy as np
from environment import BaseEnvironment


class MountainEnvironment(BaseEnvironment):

    def __init__(self):
        reward = None
        state = None
        is_terminal = None
        self.current_state = None
        self.reward_state_term = (reward, state, is_terminal)
        self.min_position = None
        self.max_position = None
        self.min_velocity = None
        self.max_velocity = None
        self.gravity = None
        self.action_discount = None

    def env_init(self, env_info={}):
        '''
        Assume env_info dict conatins:
        {
            min_posiiton: -1.2[int],
            max_position: 0.5[int],
            min_velocity: -0.07[int],
            max_velosity: 0.07[int],
            gravity: 0.0025[int],
            action_discount: 0.001[int],
            seed: [int]
        }
        '''
        # set random seed for each run
        # self.rand_generator = np.random.RandomState(env_info.get("seed"))
        
        # set each class attribute
        self.min_position = env_info["min_position"]
        self.max_position = env_info["max_position"]
        self.min_velocity = env_info["min_velocity"]
        self.max_velocity = env_info["max_velocity"]
        self.gravity = env_info["gravity"]
        self.action_discount = env_info["action_discount"]
        local_state = 0
        self.reward_state_term = (0.0, local_state, False)
         
    def env_start(self):
        # set starting reward, position, and velocity
        # reward = 0.0
        position = np.random.uniform(-0.6, -0.4)
        velocity = 0.0
        # set position and velocity to a tuple called state
        self.current_state = np.array([position, velocity])

        return self.current_state

    def env_step(self, agent_action):
        # set last_postion and last_velocity from self.reward_state_term
        last_position, last_velocity = self.current_state
        # updated action
        if agent_action == 0:
            action = -1
        elif agent_action == 1:
            action = 0
        elif agent_action == 2:
            action = 1

        # updated velocity
        velocity = last_velocity + self.action_discount * action - self.gravity * np.cos(3 * last_position)
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
            # set current_state = None
        # checked position within environment bounds 
        elif position < self.min_position:
            # checked if velocity needs to be reset
            velocity = 0
            position = self.min_position
            reward = -1
            is_terminal = False
        # position in bounds
        else:
            reward = -1
            is_terminal = False

        # updated self.reward_state_term
        self.current_state = np.array([position, velocity])
        self.reward_state_term = (reward, self.current_state, is_terminal)

        return self.reward_state_term
    
    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "What is the current reward?":
            return "{}".format(self.reward_state_term[0])
        else:
            return "No idea"