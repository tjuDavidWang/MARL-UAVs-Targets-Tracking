import numpy as np
import math


dt = 0.1


class UAV:
    def __init__(self, x0, y0, h0, vmax=10, hmax=np.pi/4, Na=5, dc=20, dp=200):
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
        self.observation = {}

    def distance(self, target):
        '''
        calculate the distance from uav to target
        :param uav:
        :param target:
        :return: scalar
        '''
        return np.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    def discrete_action(self, a_idx):
        '''
        from the action space index to the real difference
        :param a_idx:
        :return: action
        '''
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.hmax / (self.Na - 1)

    def update_position(self, a_idx):
        a = self.discrete_action(a_idx)
        dx = dt * self.vmax * np.cos(self.h)  # x 方向位移
        dy = dt * self.vmax * np.sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        self.h += dt * a  # 更新朝向角度
        self.h = (self.h + np.pi) % (2 * np.pi) - np.pi  # 确保朝向角度在 [-pi, pi) 范围内
        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)

    def observe_target(self, targets_list):
        self.observation = {}  # Reset observed targets
        for target in targets_list:
            dist = self.distance(target)
            if dist <= self.dp:
                self.observation[target] = dist
            else:
                self.observation[target] = -1  # Not observed but within perception range

    def calculate_multi_target_tracking_reward(self):
        track_reward = 0
        for target, distance in self.observation.items():
            if distance <= self.dp:
                track_reward += 1 + (self.dp - distance) / self.dp
        return track_reward

    def calculate_duplicate_tracking_punishment(self, uavs, radio=2):  # radio用来控制惩罚的范围
        total_punishment = 0
        for other_uav in uavs:
            if other_uav != self:
                distance = self.distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * math.exp((radio * self.dp - distance) / (radio * self.dp))
                    total_punishment += punishment
        return total_punishment

    def calculate_boundary_punishment(self, xmax, ymax, dmin):
        boundary_punishment = 0
        if self.x < dmin or self.x > (xmax - dmin) or self.y < dmin or self.y > (ymax - min):
            boundary_punishment = -0.5 * (self.dp - dmin) / self.dp
        else:
            boundary_punishment = 0
        return boundary_punishment
