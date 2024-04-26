import numpy as np
from math import cos, sin, sqrt, exp, pi
from typing import List, Tuple
from PMINet import PMINetwork

dt = 0.1


class UAV:
    def __init__(self, x0: float, y0: float, h0: float, a_idx: int,
                 v_max: float = 100, h_max: float = pi / 6, na: int = 5, dc: float = 500, dp: float = 200):
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

        # action
        # self.a = np.zeros(na)
        # self.a[a_idx] = 1
        self.a = a_idx

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp

        # set of local information
        # self.communication = []
        self.target_observation = []
        self.uav_communication = []

        # reward
        self.raw_reward = 0
        self.reward = 0

    def __distance(self, target: 'UAV') -> float:
        """
        calculate the distance from uav to target
        :param target: class UAV
        :return: scalar
        """
        return sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    @staticmethod
    def distance(x1, y1, x2, y2) -> float:
        """
        calculate the distance from uav to target
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def discrete_action(self, a_idx: int) -> float:  # 命名不太规范, 之后修改
        """
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        """
        # from action space to the real world action
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    # def discrete_action(self, action: 'np.ndarray') -> float:  # 命名不太规范, 之后修改
    #     """
    #     from the action space index to the real difference
    #     :param action: {0,1,...,Na - 1}
    #     :return: action : scalar 即角度改变量
    #     """
    #     # from action space to the real world action
    #     na = np.argmax(action).item() + 1  # 从 1 开始索引
    #     return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: 'int') -> (float, float, float):
        """
        receive the index from action space, then update the current position
        :param action: {0,1,...,Na - 1}
        :return:
        """
        self.a = action
        a = self.discrete_action(action)  # 有可能把这行放到其他位置

        dx = dt * self.v_max * cos(self.h)  # x 方向位移
        dy = dt * self.v_max * sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        self.h += dt * a  # 更新朝向角度
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内

        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)

    def observe_target(self, targets_list: List['UAV'], relative=False):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return:
        """
        self.target_observation = []  # Reset observed targets
        for target in targets_list:
            dist = self.__distance(target)
            if dist <= self.dp:
                # add (x, y, vx, vy) information
                if relative:
                    self.target_observation.append((target.x - self.x,
                                                    target.y - self.x,
                                                    cos(target.h) * target.v_max - cos(self.h) * self.v_max,
                                                    sin(target.h) * target.v_max - sin(self.h) * self.v_max))
                else:
                    self.target_observation.append((target.x,
                                                    target.y,
                                                    cos(target.h) * target.v_max,
                                                    sin(target.h) * target.v_max))

    # def __observe_target_with_fixed_size(self, targets_list: List['UAV']):
    #     """
    #     Observing target with a radius within dp
    #     :param targets_list: [class UAV]
    #     :return:
    #     """
    #     self.target_observation = []  # Reset observed targets
    #     for target in targets_list:
    #         dist = self.__distance(target)
    #         if dist <= self.dp:
    #             # add (x, y, vx, vy) information
    #             self.target_observation.append((target.x,
    #                                             target.y,
    #                                             cos(target.h) * target.v_max,
    #                                             sin(target.h) * target.v_max))
    #         else:
    #             self.target_observation.append((0, 0, 0, 0))  # TODO, 值不规范

    # def __observe_uav_with_fixed_size(self, uav_list: List['UAV']):  # communication
    #     """
    #     communicate with other uav_s with a radius within dp
    #     :param uav_list: [class UAV]
    #     :return:
    #     """
    #     self.uav_communication = []  # Reset observed targets
    #     for uav in uav_list:
    #         dist = self.__distance(uav)
    #         if dist <= self.dc and uav != self:
    #             # add (x, y, vx, vy, a) information
    #             self.uav_communication.append((uav.x,
    #                                            uav.y,
    #                                            cos(uav.h) * uav.v_max,
    #                                            sin(uav.h) * uav.v_max,
    #                                            uav.a))
    #         else:
    #             self.uav_communication.append((0, 0, 0, 0, np.zeros(self.Na)))  # TODO, 值不规范, 且a已改为np数组

    def observe_uav(self, uav_list: List['UAV'], relative=False):  # communication
        """
        communicate with other uav_s with a radius within dp
        :param relative: relative to uav itself
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_communication = []  # Reset observed targets
        for uav in uav_list:
            dist = self.__distance(uav)
            if dist <= self.dc and uav != self:
                # add (x, y, vx, vy, a) information
                if relative:
                    self.uav_communication.append((uav.x - self.x,
                                                   uav.y - self.y,
                                                   cos(uav.h) * uav.v_max - cos(self.h) * self.v_max,
                                                   sin(uav.h) * uav.v_max - sin(self.h) * self.v_max,
                                                   uav.a - self.a))
                else:
                    self.uav_communication.append((uav.x,
                                                   uav.y,
                                                   cos(uav.h) * uav.v_max,
                                                   sin(uav.h) * uav.v_max,
                                                   uav.a))

    # @staticmethod
    # def __transform_to_array2d(data: List[Tuple[float, float, float, float, 'np.ndarray']]) -> 'np.ndarray':
    #     elements = [list(elem)[:-1] for elem in data]
    #
    #     # 获取数组内容并添加到提取的元素列表中
    #     for elem in data:
    #         elements[data.index(elem)].extend(list(elem[-1]))
    #
    #     # 将元素组合成二维数组
    #     array_2d = np.array(elements)
    #     return array_2d
    #
    # @staticmethod
    # def __transform_to_array1d(data: Tuple[float, float, float, float, np.ndarray]) -> np.ndarray:
    #     array_1d = np.hstack((np.array(data[:-1]), data[-1]))
    #     return array_1d

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, int]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, int]):
        """
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        """
        return self.uav_communication, self.target_observation, (self.x, self.y, self.a)

    # def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, 'np.ndarray']],
    #                                     List[Tuple[float, float, float, float]], Tuple[float, float, 'np.ndarray']):
    #     """
    #     :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
    #     """
    #     return self.uav_communication, self.target_observation, (self.x, self.y, self.a)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, sb = self.__get_all_local_state()

        if communication:
            d_communication = []  # store the distance from each uav to itself
            for x, y, vx, vy, na in communication:
                d_communication.append(self.distance(x, y, self.x, self.y))

            # regularization by the distance
            # communication = self.__transform_to_array2d(communication)
            communication = np.array(communication)
            communication_weighted = communication / np.array(d_communication)[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=0)
        else:
            # average_communication = np.zeros(4 + self.Na)  # empty communication
            average_communication = -np.ones(4 + 1)  # empty communication  # TODO -1合法吗

        if observation:
            d_observation = []  # store the distance from each target to itself
            for x, y, vx, vy in observation:
                d_observation.append(self.distance(x, y, self.x, self.y))

            # regularization by the distance
            observation = np.array(observation)
            observation_weighted = observation / np.array(d_observation)[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation  # TODO -1合法吗

        # sb = self.__transform_to_array1d(sb)
        sb = np.array(sb)
        result = np.hstack((average_communication, average_observation, sb))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        :return: np.ndarray
        """
        # using weighted mean method:    dim:
        return self.__get_local_state_by_weighted_mean()

    def __calculate_multi_target_tracking_reward(self) -> float:
        """
        calculate multi target tracking reward
        :return: scalar
        """
        track_reward = 0
        for x, y, _, _ in self.target_observation:
            distance = self.distance(x, y, self.x, self.y)
            track_reward += 1 + (self.dp - distance) / self.dp
        return track_reward

    def __calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=2) -> float:
        """
        calculate duplicate tracking punishment
        :param uav_list: [class UAV]
        :param radio: radio用来控制惩罚的范围, 超出多远才算入惩罚
        :return: scalar
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
                    total_punishment += punishment
        return total_punishment

    def __calculate_boundary_punishment(self, x_max: float, y_max: float) -> float:
        """
        :param x_max: border of the map at x-axis, scalar
        :param y_max: border of the map at y-axis, scalar
        :return:
        """
        x_to_0 = self.x - 0
        x_to_max = x_max - self.x
        y_to_0 = self.x - 0
        y_to_max = y_max - self.y
        d_bdr = min(x_to_0, x_to_max, y_to_0, y_to_max)
        if 0 <= self.x <= x_max and 0 <= self.y <= y_max:
            if d_bdr < self.dp:
                boundary_punishment = -0.5 * (self.dp - d_bdr) / self.dp
            else:
                boundary_punishment = 0
        else:
            boundary_punishment = 10 * d_bdr / self.dp
        return boundary_punishment

    def calculate_raw_reward(self, uav_list: List['UAV'], x_max, y_max):
        """
        calculate three parts of the reward/punishment for each uav
        :return:
        """
        self.raw_reward = 0
        for uav in uav_list:
            if uav != self:
                reward = uav.__calculate_multi_target_tracking_reward()
                boundary_punishment = uav.__calculate_boundary_punishment(x_max, y_max)
                punishment = uav.__calculate_duplicate_tracking_punishment(uav_list)
                self.raw_reward += reward + boundary_punishment + punishment

    def calculate_cooperative_reward(self, pmi_net: 'PMINetwork', uav_list: List['UAV'], a=0.5) -> float:
        """
        calculate cooperative reward
        :param pmi_net:
        :param uav_list: [class UAV]
        :param a:
        :return:
        """
        neighbor_rewards = []
        neighbor_dependencies = []
        la = self.get_local_state()

        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
                other_uav_la = other_uav.get_local_state()
                # _input = np.hstack((la, other_uav_la))
                _input = la * other_uav_la
                neighbor_dependencies.append(pmi_net.inference(_input.squeeze()))

        neighbor_rewards = np.array(neighbor_rewards)
        neighbor_dependencies = np.array(neighbor_dependencies)
        self.reward = (1 - a) * self.raw_reward + a * np.sum(neighbor_rewards * neighbor_dependencies).item()
        return self.reward
