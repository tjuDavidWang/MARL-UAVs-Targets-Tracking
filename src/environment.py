from uav import UAV
import numpy as np
import torch
import math


class Environment:
    def __init__(self, n_uavs=3, m_targets=5, xmax=2000, ymax=2000, dmin=5):
        # size of the environment
        self.dmin = dmin
        self.xmax = xmax
        self.ymax = ymax

        # dim of action space and state space
        self.state_dim = (n_uavs + m_targets) * 4  # x, y, vx, vy,
        self.action_dim = (n_uavs + m_targets)

        # agents in the environments
        self.n_uavs = n_uavs
        self.m_targets = m_targets
        self.reset()

# 重置环境
    def reset(self):
        '''
        reset the location for all uavs at (0, 0)
        :return: should be the initial states !!!!
        '''
        self.uavs = [UAV(0, 0, np.random.uniform(-np.pi, np.pi)) for _ in range(self.n_uavs)]
        self.targets = [UAV(np.random.uniform(self.xmax), np.random.uniform(self.ymax), np.random.uniform(-np.pi, np.pi))
                        for _ in range(self.m_targets)]
        uav_states, target_states = self.get_states()
        return np.concatenate((uav_states, target_states)).flatten()

    def get_states(self):
        '''
        get the state of the uavs and targets
        :return: list of tuples
        '''
        uav_states = []
        target_states = []
        for uav in self.uavs:
            uav_states.append((uav.x, uav.y, uav.vmax * np.cos(uav.h), uav.vmax * np.sin(uav.h)))
        for target in self.targets:
            target_states.append((target.x, target.y, target.vmax * np.cos(target.h), target.vmax * np.sin(target.h)))
        return uav_states, target_states

    def step(self, actions):
        '''
        state transfer functions
        :param actions:
        :return: states, rewards, done, tem
        '''
        for i, uav in enumerate(self.uavs):
            uav.update_position(actions[i])
            uav.observe_target(self.targets)

        rewards = self.calculate_rewards()
        states = self.get_states()
        done = self.check_done()
        tem = []
        return states, rewards, done, tem

    # def observe_target(self, targets_list):
    #     for uav in self.uavs:
    #         uav.observation = {}  # Reset observed targets
    #         for target in targets_list:
    #             dist = self.distance(uav, target)
    #             if dist <= uav.dp:
    #                 uav.observation[target] = dist
    #             else:
    #                 uav.observation[target] = -1  # Not observed but within perception range

    def calculate_raw_reward(self):
        '''
        not consider reciprocal record
        :return:
        '''
        for uav in self.uavs:
            reward = uav.calculate_multi_target_tracking_reward()
            boundary_punishment = uav.calculate_boundary_punishment(self.xmax, self.ymax, self.dmin)
            punishment = uav.calculate_duplicate_tracking_punishment(self.uavs)
            uav.raw_reward = reward + boundary_punishment + punishment

    # def calculate_cooperative_reward(self, a=0.5):  # a:互惠程度, 0表示不考虑互惠
    #     for uav in self.uavs:
    #         neighbor_rewards = []
    #         for other_uav in self.uavs:
    #             if other_uav != self and self.distance(uav, other_uav) <= uav.search_radius:
    #                 neighbor_rewards.append(other_uav.raw_reward)
    #
    #         uav.reward = (1 - a) * uav.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards)
