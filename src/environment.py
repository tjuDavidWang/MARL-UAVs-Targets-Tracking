from uav import UAV
import numpy as np
import torch
import math


class Environment:
    def __init__(self, n_uavs=3, m_targets=5, xmax=2000, ymax=2000, dmin=5, Na=5):
        '''
        :param n_uavs: scalar
        :param m_targets: scalar
        :param xmax: scalar
        :param ymax: scalar
        :param dmin: scalar
        :param Na: scalar
        '''
        # size of the environment
        self.dmin = dmin
        self.xmax = xmax
        self.ymax = ymax

        # dim of action space and state space
        self.state_dim = 3  # x, y, h/theta
        self.action_dim = Na

        # agents in the environments
        self.n_uavs = n_uavs
        self.m_targets = m_targets
        self.reset()

    def reset(self):
        '''
        reset the location for all uavs at (0, 0)
        :return: should be the initial states !!!!
        '''
        self.uavs = [UAV(0, 0, np.random.uniform(-np.pi, np.pi)) for _ in range(self.n_uavs)]
        self.targets = [UAV(np.random.uniform(self.xmax), np.random.uniform(self.ymax), np.random.uniform(-np.pi, np.pi))
                        for _ in range(self.m_targets)]

    def get_states(self):
        '''
        get the state of the uavs and targets
        :return: [[(x, y, vx, by, na),...]] for all uavs, [[(x, y, vx, vy)]] for all targets
        '''
        uav_states = []
        target_states = []
        for uav in self.uavs:
            uav_states.append(uav.uav_observation)
        for target in self.targets:
            target_states.append(target.target_observation)
        return uav_states, target_states

    def step(self, actions):
        '''
        state transfer functions
        :param actions: {0,1,...,Na - 1}
        :return: states, rewards  // TODO
        '''
        for i, uav in enumerate(self.uavs):
            uav.update_position(actions[i])
            uav.observe_target(self.targets)
            uav.observe_uav(self.uavs)

        rewards = self.calculate_rewards()
        next_states = self.get_states()
        # done = self.check_done()
        # tem = []
        return next_states, rewards

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

    def calculate_rewards(self):
        rewards = []
        for uav in self.uavs:
            uav.reward = uav.calculate_cooperative_reward(self.uavs)
            rewards.append(uav.reward)
        return rewards
