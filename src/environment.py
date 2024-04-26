from uav import UAV
from target import TARGET
import numpy as np
from math import pi
import random
from typing import List


class Environment:
    def __init__(self, n_uav: int = 3, m_targets: int = 5, x_max: float = 2000, y_max: float = 2000,
                 d_min: float = 50, na: int = 5):
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

        # position of uav and target
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

    def reset(self):
        """
        reset the location for all uav_s at (self.d_min, self.d_min)
        :return: should be the initial states !!!!
        """
        # the initial position of the uav is (self.d_min, self.d_min), having randon headings
        self.uav_list = [UAV(self.d_min, self.d_min, random.uniform(-pi, pi),
                             random.randint(0, self.action_dim - 1)) for _ in range(self.n_uav)]
        # the initial position of the uav is (self.d_min, self.d_min), having randon headings
        # self.uav_list = [UAV(0, 0, random.uniform(-pi, pi),
        #                      random.randint(0, self.action_dim - 1)) for _ in range(self.n_uav)]

        # the initial position of the target is random, having randon headings
        # self.target_list = [UAV(random.uniform(0, self.x_max),
        #                         random.uniform(0, self.y_max),
        #                         random.uniform(-pi, pi),
        #                         random.randint(0, self.action_dim - 1))  # TODO, 目标可以连续移动
        #                     for _ in range(self.m_targets)]
        self.target_list = [TARGET(random.uniform(0, self.x_max),
                                   random.uniform(0, self.y_max),
                                   random.uniform(-pi, pi),
                                   random.uniform(-pi/6, pi/6))  # TODO, 目标可以连续移动
                            for _ in range(self.m_targets)]
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

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
        for i, target in enumerate(self.target_list):
            target.update_position()

        for i, uav in enumerate(self.uav_list):
            uav.update_position(actions[i])
            uav.observe_target(self.target_list)
            uav.observe_uav(self.uav_list)

        rewards = self.calculate_rewards()
        next_states = self.get_states()

        # trace the position matrix
        target_xs, target_ys = self.__get_all_target_position()
        self.position['all_target_xs'].append(target_xs)
        self.position['all_target_ys'].append(target_ys)
        uav_xs, uav_ys = self.__get_all_uav_position()
        self.position['all_uav_xs'].append(uav_xs)
        self.position['all_uav_ys'].append(uav_ys)

        return next_states, rewards

    def __get_all_uav_position(self) -> (List[float], List[float]):
        uav_xs = []
        uav_ys = []
        for uav in self.uav_list:
            uav_xs.append(uav.x)
            uav_ys.append(uav.y)
        return uav_xs, uav_ys

    def __get_all_target_position(self) -> (List[float], List[float]):
        target_xs = []
        target_ys = []
        for target in self.target_list:
            target_xs.append(target.x)
            target_ys.append(target.y)
        return target_xs, target_ys

    def get_uav_and_target_position(self) -> (List[float], List[float], List[float], List[float]):
        return (self.position['all_uav_xs'], self.position['all_uav_ys'],
                self.position['all_target_xs'], self.position['all_target_ys'])

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
