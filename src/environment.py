import os.path

from agent.uav import UAV
from agent.target import TARGET
import numpy as np
from math import pi
import random
from typing import List


class Environment:
    def __init__(self, n_uav: int, m_targets: int, x_max: float, y_max: float, na: int):
        """
        :param n_uav: scalar
        :param m_targets: scalar
        :param x_max: scalar
        :param y_max: scalar
        :param na: scalar
        """
        # size of the environment
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

        # position of uav and target
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

    def __reset(self, t_v_max, t_h_max, u_v_max, u_h_max, na, dc, dp, dt, init_x, init_y):
        """
        reset the location for all uav_s at (init_x, init_y)
        reset the store position to empty
        :return: should be the initial states !!!!
        """
        # the initial position of the uav is (self.d_min, self.d_min), having randon headings
        self.uav_list = [UAV(init_x,
                             init_y,
                             random.uniform(-pi, pi),
                             random.randint(0, self.action_dim - 1),
                             u_v_max, u_h_max, na, dc, dp, dt) for _ in range(self.n_uav)]

        # the initial position of the target is random, having randon headings
        self.target_list = [TARGET(random.uniform(0, self.x_max),
                                   random.uniform(0, self.y_max),
                                   random.uniform(-pi, pi),
                                   random.uniform(-pi / 6, pi / 6),
                                   t_v_max, t_h_max, dt)
                            for _ in range(self.m_targets)]
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': []}

    def reset(self, config):
        self.__reset(t_v_max=config["target"]["v_max"],
                     t_h_max=pi / float(config["target"]["h_max"]),
                     u_v_max=config["uav"]["v_max"],
                     u_h_max=pi / float(config["uav"]["h_max"]),
                     na=config["environment"]["na"],
                     dc=config["uav"]["dc"],
                     dp=config["uav"]["dp"],
                     dt=config["uav"]["dt"],
                     init_x=200, init_y=200)

    def get_states(self) -> (List['np.ndarray']):
        """
        get the state of the uav_s
        :return: list of np array, each element is a 1-dim array with size of 12
        """
        uav_states = []
        # collect the overall communication and target observation by each uav
        for uav in self.uav_list:
            uav_states.append(uav.get_local_state())
        return uav_states

    def step(self, config, pmi, actions):
        """
        state transfer functions
        :param config:
        :param pmi: PMI network
        :param actions: {0,1,...,Na - 1}
        :return: states, rewards
        """
        # update the position of targets
        for i, target in enumerate(self.target_list):
            target.update_position(self.x_max, self.y_max)

        # update the position of targets
        for i, uav in enumerate(self.uav_list):
            uav.update_position(actions[i])

            # observation and communication
            uav.observe_target(self.target_list)
            uav.observe_uav(self.uav_list)

        (rewards,
         target_tracking_reward,
         boundary_punishment,
         duplicate_tracking_punishment) = self.calculate_rewards(config=config, pmi=pmi)
        next_states = self.get_states()

        # trace the position matrix
        target_xs, target_ys = self.__get_all_target_position()
        self.position['all_target_xs'].append(target_xs)
        self.position['all_target_ys'].append(target_ys)
        uav_xs, uav_ys = self.__get_all_uav_position()
        self.position['all_uav_xs'].append(uav_xs)
        self.position['all_uav_ys'].append(uav_ys)

        reward = {
            'rewards': rewards,
            'target_tracking_reward': target_tracking_reward,
            'boundary_punishment': boundary_punishment,
            'duplicate_tracking_punishment': duplicate_tracking_punishment
        }

        return next_states, reward

    def __get_all_uav_position(self) -> (List[float], List[float]):
        """
        :return: all the position of the uav through this epoch
        """
        uav_xs = []
        uav_ys = []
        for uav in self.uav_list:
            uav_xs.append(uav.x)
            uav_ys.append(uav.y)
        return uav_xs, uav_ys

    def __get_all_target_position(self) -> (List[float], List[float]):
        """
        :return: all the position of the targets through this epoch
        """
        target_xs = []
        target_ys = []
        for target in self.target_list:
            target_xs.append(target.x)
            target_ys.append(target.y)
        return target_xs, target_ys

    def get_uav_and_target_position(self) -> (List[float], List[float], List[float], List[float]):
        """
        :return: both the uav and the target position matrix
        """
        return (self.position['all_uav_xs'], self.position['all_uav_ys'],
                self.position['all_target_xs'], self.position['all_target_ys'])

    def calculate_rewards(self, config, pmi) -> ([float], float, float, float):
        # raw reward first
        target_tracking_rewards = []
        boundary_punishments = []
        duplicate_tracking_punishments = []
        for uav in self.uav_list:
            (target_tracking_reward,
             boundary_punishment,
             duplicate_tracking_punishment) = uav.calculate_raw_reward(self.uav_list, self.x_max, self.y_max,
                                                                       config["uav"]["gamma"], config["uav"]["gamma"],
                                                                       config["uav"]["gamma"])

            target_tracking_rewards.append(target_tracking_reward)
            boundary_punishments.append(boundary_punishment)
            duplicate_tracking_punishments.append(duplicate_tracking_punishment)

        rewards = []
        for uav in self.uav_list:
            uav.calculate_cooperative_reward(self.uav_list, pmi)
            rewards.append(uav.reward)
        return rewards, target_tracking_rewards, boundary_punishments, duplicate_tracking_punishments

    def save_position(self, save_dir, epoch_i):
        u_xy = np.array([self.position["all_uav_xs"],
                         self.position["all_uav_ys"]]).transpose()  # n_uav * num_steps * 2
        t_xy = np.array([self.position["all_target_xs"],
                         self.position["all_target_ys"]]).transpose()  # m_target * num_steps * 2

        np.savetxt(os.path.join(save_dir, "u_xy", 'u_xy' + str(epoch_i) + '.csv'),
                   u_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
        np.savetxt(os.path.join(save_dir, "t_xy", 't_xy' + str(epoch_i) + '.csv'),
                   t_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
