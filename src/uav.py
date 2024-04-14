import numpy as np
import math


dt = 0.1


class UAV:
    def __init__(self, x0, y0, h0, vmax=10, hmax=np.pi/4, Na=5, dc=20, dp=200):
        '''
        :param x0: scalar
        :param y0: scalar
        :param h0: scalar
        :param vmax: scalar
        :param hmax: scalar
        :param Na: scalar
        :param dc: scalar
        :param dp: scalar
        '''
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.vmax = vmax

        # the max heading angular rate and the action of this uav
        self.hmax = hmax
        self.Na = Na
        self.na = Na - 1

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp

        # set of local information
        self.communication = {}
        self.target_observation = {}
        self.uav_observation = {}

        # reward
        self.raw_reward = 0
        self.reward = 0

    def distance(self, target):
        '''
        calculate the distance from uav to target
        :param uav:
        :param target: class UAV
        :return: scalar
        '''
        return np.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    def discrete_action(self, a_idx):
        '''
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        '''
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.hmax / (self.Na - 1)

    def update_position(self, a_idx):
        '''
        receive the index from action space, then update the current position
        :param a_idx: {0,1,...,Na - 1}
        :return:
        '''
        self.na = self.discrete_action(a_idx)
        dx = dt * self.vmax * np.cos(self.h)  # x 方向位移
        dy = dt * self.vmax * np.sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        self.h += dt * self.na  # 更新朝向角度
        self.h = (self.h + np.pi) % (2 * np.pi) - np.pi  # 确保朝向角度在 [-pi, pi) 范围内
        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)

    def observe_target(self, targets_list):
        '''
        Observing target with a radius within dp
        :param targets_list: [class UAV]
        :return:
        '''
        self.target_observation = []  # Reset observed targets
        for target in targets_list:
            dist = self.distance(target)
            if dist <= self.dp:
                self.target_observation.append((target.x, target.y, np.cos(target.h) * target.vmax, np.sin(target.h) * target.vmax))
            else:
                self.target_observation.append((0, 0, 0, 0))  # Not observed but within perception range

    def observe_uav(self, uavs_list):  # communication
        '''
        communicate with other uavs with a radius within dp
        :param uavs_list: [class UAV]
        :return:
        '''
        self.uav_observation = []  # Reset observed targets
        for uav in uavs_list:
            dist = self.distance(uav)
            if dist <= self.dc:
                self.uav_observation.append((uav.x, uav.y, np.cos(uav.h) * uav.vmax, np.sin(uav.h) * uav.vmax, uav.na))
            else:
                self.uav_observation.append((0, 0, 0, 0, 0))  # Not observed but within perception range

    def get_local_state(self):
        '''
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        '''
        return self.uav_observation, self.target_observation, (self.x, self.y, self.na)

    def calculate_multi_target_tracking_reward(self):
        '''
        calculate multi target tracking reward
        :return: scalar
        '''
        track_reward = 0
        for target, distance in self.observation.items():
            if distance <= self.dp:
                track_reward += 1 + (self.dp - distance) / self.dp
        return track_reward

    def calculate_duplicate_tracking_punishment(self, uavs, radio=2):
        '''
        calculate duplicate tracking punishment
        :param uavs: [class UAV]
        :param radio: radio用来控制惩罚的范围, 超出多远才算入惩罚
        :return: scalar
        '''
        total_punishment = 0
        for other_uav in uavs:
            if other_uav != self:
                distance = self.distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * math.exp((radio * self.dp - distance) / (radio * self.dp))
                    total_punishment += punishment
        return total_punishment

    def calculate_boundary_punishment(self, xmax, ymax, dmin):
        '''

        :param xmax: border of the map at x-axis, scalar
        :param ymax: border of the map at y-axis, scalar
        :param dmin: minimum distance to the border, scalar
        :return: scalar
        '''
        if self.x < dmin or self.x > (xmax - dmin) or self.y < dmin or self.y > (ymax - min):
            boundary_punishment = -0.5 * (self.dp - dmin) / self.dp
        else:
            boundary_punishment = 0
        return boundary_punishment
    
    def calculate_cooperative_reward(self, uavs, a=0.5):
        '''
        calculate cooperative reward
        :param uavs: [class UAV]
        :param a:
        :return:
        '''
        neighbor_rewards = []
        for other_uav in uavs:
            if other_uav != self and self.distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)

        self.reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards)
