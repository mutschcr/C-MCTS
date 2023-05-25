"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import copy


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    Description:
        An agent moves inside of a small maze.
    Source:
        This environment is a variation on common grid world environments.
    Observation:
        Type: Box(1)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    metadata = {
        'render.modes': ['ascii'],
    }

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 3]))  # Creates a cartesian product: list of two tuples -> (rowindex, colindex)

        self.agent_position = [0, 0]  # Top left tile (S)
        # END

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Action is an integer or a one-dimensional np.ndarray.
        # 0: Up
        # 1: Right
        # 2: Down
        # 3: Left

        if 0 == action and self.agent_position[0] - 1 >= 0:  # Up
            self.agent_position[0] -= 1
        elif 1 == action and self.agent_position[1] + 1 < 4:  # Right
            self.agent_position[1] += 1
        elif 2 == action and self.agent_position[0] + 1 < 4:  # Down
            self.agent_position[0] += 1
        elif 3 == action and self.agent_position[1] - 1 >= 0:  # Left
            self.agent_position[1] -= 1

        observation = self.agent_position
        cellmark = self.map[observation[0]][observation[1]]
        if cellmark == 'g':
            done = True
            reward = 10
        elif cellmark == 't':
            done = True
            reward = -10
        else:
            done = False
            reward = -1

        return observation, reward, done, {}

    def set(self, state):
        self.agent_position = state

    def reset(self):
        self.agent_position = [0, 0]
        return self.agent_position

    def observe(self):
        """Helper-method that creates an observation. Useful, but not required or used by gym."""
        return None

    def render(self, mode='ascii'):
        render = copy.deepcopy(self.map)
        render[self.agent_position[0]][self.agent_position[1]] = '*'
        for i in range(4):
            print(render[i])
        print("")
        return None

    def close(self):
        pass
