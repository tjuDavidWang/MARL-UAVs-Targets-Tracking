from uav import UAV
import numpy as np
from math import pi
import random
from typing import List


class Environment:
    def __init__(self, n_uav: int = 3, m_targets: int = 5, x_max: float = 2000, y_max: float = 2000,
                 d_min: float = 5, na: int = 5):
        """
        :param n_uav: scalar
        :param m_targets: scalar
        :param x_max: scalar
        :param y_max: scalar
        :param d_min: scalar
        :param na: scalar
        """
        # size of the environment
        self.d_min = d_min
        self.x_max = x_max
        self.y_max = y_max

        # dim of action space and state space
        # communication(4 scalar, a), observation(4 scalar), boundary and state information(2 scalar, a)
        # self.state_dim = (4 + na) + 4 + (2 + na)
        self.state_dim = (4 + 1) + 4 + (2 + 1)
        self.action_dim = na

        # agents parameters in the environments
        self.n_uav = n_uav
        self.m_targets = m_targets

        # agents
        self.uav_list = []
        self.target_list = []
        self.reset()

    def reset(self):
        """
        reset the location for all uav_s at (0, 0)
        :return: should be the initial states !!!!
        """
        # the initial position of the uav is (0, 0), having randon headings
        self.uav_list = [UAV(0, 0, random.uniform(-pi, pi),
                             random.randint(0, self.action_dim - 1)) for _ in range(self.n_uav)]

        # the initial position of the target is random, having randon headings
        self.target_list = [UAV(random.uniform(0, self.x_max),
                                random.uniform(0, self.y_max),
                                random.uniform(-pi, pi),
                                random.randint(0, self.action_dim - 1))  # TODO, 目标可以连续移动
                            for _ in range(self.m_targets)]

    def get_states(self) -> (List['np.ndarray']):
        """
        get the state of the uav_s
        :return: list of np array, each element is a 1-dim array with size of 12
        """
        uav_states = []

        # collect the overall communication and target observation by each uav
        for uav in self.uav_list:
            uav_states.append(uav.get_local_state())

        # global state of target, maybe not used
        # for target in self.target_list:
        #     target_states.append(target.target_observation)
        return uav_states

    def step(self, actions):
        """
        state transfer functions
        :param actions: {0,1,...,Na - 1}
        :return: states, rewards
        """
        for i, uav in enumerate(self.uav_list):
            uav.update_position(actions[i])
            uav.observe_target(self.target_list)
            uav.observe_uav(self.uav_list)

        rewards = self.calculate_rewards()
        next_states = self.get_states()
        # done = self.check_done()
        # tem = []
        return next_states, rewards

    # def observe_target(self, targets_list):
    #     for uav in self.uav_s:
    #         uav.observation = {}  # Reset observed targets
    #         for target in targets_list:
    #             dist = self.distance(uav, target)
    #             if dist <= uav.dp:
    #                 uav.observation[target] = dist
    #             else:
    #                 uav.observation[target] = -1  # Not observed but within perception range

    def calculate_rewards(self) -> [float]:
        # raw reward first
        for uav in self.uav_list:
            uav.calculate_raw_reward(self.uav_list, self.x_max, self.y_max, self.d_min)

        rewards = []
        for uav in self.uav_list:
            uav.reward = uav.calculate_cooperative_reward(self.uav_list)
            rewards.append(uav.reward)
        return rewards
