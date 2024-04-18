from uav import UAV
import numpy as np
# from typing import List


class Environment:
    def __init__(self, n_uav=3, m_targets=5, x_max=2000, y_max=2000, d_min=5, na=5):
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
        self.state_dim = 12  # 5 of communication, 4 of observation, 3 of boundary and state information
        self.action_dim = na

        # agents in the environments
        self.n_uav = n_uav
        self.m_targets = m_targets
        self.reset()

        # agents
        self.uav_list = []
        self.target_list = []

    def reset(self):
        """
        reset the location for all uav_s at (0, 0)
        :return: should be the initial states !!!!
        """
        self.uav_list = [UAV(0, 0, np.random.uniform(-np.pi, np.pi)) for _ in range(self.n_uav)]
        self.target_list = [UAV(np.random.uniform(self.x_max),
                                np.random.uniform(self.y_max),
                                np.random.uniform(-np.pi, np.pi))
                            for _ in range(self.m_targets)]

    def get_states(self) -> ([[(float, float, float, float, float)]], [[(float, float, float, float)]]):
        """
        get the state of the uav_s and targets
        :return: [[(x, y, vx, by, na),...]] for all uav_s, [[(x, y, vx, vy)]] for all targets
        """
        uav_states = []
        target_states = []
        for uav in self.uav_list:
            uav_states.append(uav.uav_observation)
        for target in self.target_list:
            target_states.append(target.target_observation)
        return uav_states, target_states

    def step(self, actions):
        """
        state transfer functions
        :param actions: {0,1,...,Na - 1}
        :return: states, rewards  // TODO
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

    def calculate_raw_reward(self):
        """
        not consider reciprocal record
        :return:
        """
        for uav in self.uav_list:
            reward = uav.calculate_multi_target_tracking_reward()
            boundary_punishment = uav.calculate_boundary_punishment(self.x_max, self.y_max, self.d_min)
            punishment = uav.calculate_duplicate_tracking_punishment(self.uav_list)
            uav.raw_reward = reward + boundary_punishment + punishment

    def calculate_rewards(self):
        rewards = []
        for uav in self.uav_list:
            uav.reward = uav.calculate_cooperative_reward(self.uav_list)
            rewards.append(uav.reward)
        return rewards
