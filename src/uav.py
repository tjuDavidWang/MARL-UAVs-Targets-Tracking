import numpy as np
import math

from typing import List

dt = 0.1


class UAV:
    def __init__(self, x0: float, y0: float, h0: float, v_max: float = 10, h_max: float = np.pi / 4,
                 na: int = 5, dc: float = 20, dp: float = 200):
        """
        :param x0: scalar
        :param y0: scalar
        :param h0: scalar
        :param v_max: scalar
        :param h_max: scalar
        :param na: scalar
        :param dc: scalar
        :param dp: scalar
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na
        self.a = na - 1

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp

        # set of local information
        self.communication = []
        self.target_observation = []
        self.uav_observation = []

        # reward
        self.raw_reward = 0
        self.reward = 0

    def distance(self, target: 'UAV') -> float:
        """
        calculate the distance from uav to target
        :param target: class UAV
        :return: scalar
        """
        return np.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    @staticmethod
    def distance_(x1, y1, x2, y2) -> float:
        """
        calculate the distance from uav to target
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def discrete_action(self, a_idx: int) -> float:
        """
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        """
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, a_idx: int) -> (float, float, float):
        """
        receive the index from action space, then update the current position
        :param a_idx: {0,1,...,Na - 1}
        :return:
        """
        self.a = self.discrete_action(a_idx)
        dx = dt * self.v_max * np.cos(self.h)  # x 方向位移
        dy = dt * self.v_max * np.sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        self.h += dt * self.a  # 更新朝向角度
        self.h = (self.h + np.pi) % (2 * np.pi) - np.pi  # 确保朝向角度在 [-pi, pi) 范围内
        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)

    def observe_target(self, targets_list: List['UAV']):
        """
        Observing target with a radius within dp
        :param targets_list: [class UAV]
        :return:
        """
        self.target_observation = []  # Reset observed targets
        for target in targets_list:
            dist = self.distance(target)
            if dist <= self.dp:
                self.target_observation.append((target.x,
                                                target.y,
                                                np.cos(target.h) * target.v_max,
                                                np.sin(target.h) * target.v_max))

    def observe_target_with_fixed_size(self, targets_list: List['UAV']):
        """
        Observing target with a radius within dp
        :param targets_list: [class UAV]
        :return:
        """
        self.target_observation = []  # Reset observed targets
        for target in targets_list:
            dist = self.distance(target)
            if dist <= self.dp:
                self.target_observation.append((target.x,
                                                target.y,
                                                np.cos(target.h) * target.v_max,
                                                np.sin(target.h) * target.v_max))
            else:
                self.target_observation.append((0, 0, 0, 0))  # Not observed but within perception range

    def observe_uav_with_fixed_size(self, uav_list: List['UAV']):  # communication
        """
        communicate with other uav_s with a radius within dp
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_observation = []  # Reset observed targets
        for uav in uav_list:
            dist = self.distance(uav)
            if dist <= self.dc:
                self.uav_observation.append((uav.x,
                                             uav.y,
                                             np.cos(uav.h) * uav.v_max,
                                             np.sin(uav.h) * uav.v_max,
                                             uav.a))
            else:
                self.uav_observation.append((0, 0, 0, 0, 0))  # Not observed but within perception range

    def observe_uav(self, uav_list: List['UAV']):  # communication
        """
        communicate with other uav_s with a radius within dp
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_observation = []  # Reset observed targets
        for uav in uav_list:
            dist = self.distance(uav)
            if dist <= self.dc:
                self.uav_observation.append((uav.x,
                                             uav.y,
                                             np.cos(uav.h) * uav.v_max,
                                             np.sin(uav.h) * uav.v_max,
                                             uav.a))

    def get_all_local_state(self) -> ([(float, float, float, float, int)],
                                      [(float, float, float, float)], (float, float, int)):
        """
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        """
        return self.uav_observation, self.target_observation, (self.x, self.y, self.a)

    def get_local_state_by_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, sb = self.get_all_local_state()  # ? * 5, ? * 4, 3

        if communication:
            d_communication = []
            for x, y, vx, vy, na in communication:
                d_communication.append(self.distance_(x, y, self.x, self.y))
            # 对 communication 中的每个值乘上 d_communication 中的倒数
            communication_weighted = np.array(communication) / np.array(d_communication)[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=1)
        else:
            average_communication = np.zeros(5)

        if observation:
            d_observation = []
            for x, y, vx, vy in observation:
                d_observation.append(self.distance_(x, y, self.x, self.y))
            # 对 observation 中的每个值乘上 d_observation 中的倒数
            observation_weighted = np.array(observation) / np.array(d_observation)[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=1)
        else:
            average_observation = np.zeros(4)

        return np.hstack((average_communication, average_observation, np.array(sb)))

    def get_local_state(self) -> 'np.ndarray':
        return self.get_local_state_by_mean()

    def calculate_multi_target_tracking_reward(self) -> float:
        """
        calculate multi target tracking reward
        :return: scalar
        """
        track_reward = 0
        for target, distance in self.target_observation:
            if distance <= self.dp:
                track_reward += 1 + (self.dp - distance) / self.dp
        return track_reward

    def calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=2) -> float:
        """
        calculate duplicate tracking punishment
        :param uav_list: [class UAV]
        :param radio: radio用来控制惩罚的范围, 超出多远才算入惩罚
        :return: scalar
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * math.exp((radio * self.dp - distance) / (radio * self.dp))
                    total_punishment += punishment
        return total_punishment

    def calculate_boundary_punishment(self, x_max: float, y_max: float, d_min: float) -> float:
        """
        :param x_max: border of the map at x-axis, scalar
        :param y_max: border of the map at y-axis, scalar
        :param d_min: minimum distance to the border, scalar
        :return:
        """
        if self.x < d_min or self.x > (x_max - d_min) or self.y < d_min or self.y > (y_max - d_min):
            boundary_punishment = -0.5 * (self.dp - d_min) / self.dp
        else:
            boundary_punishment = 0
        return boundary_punishment
    
    def calculate_cooperative_reward(self, uav_list: List['UAV'], a=0.5):
        """
        calculate cooperative reward
        :param uav_list: [class UAV]
        :param a:
        :return:
        """
        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)

        self.reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards)
