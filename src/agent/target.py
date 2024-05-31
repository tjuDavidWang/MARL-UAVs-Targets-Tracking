from math import cos, sin, pi
import random


class TARGET:
    def __init__(self, x0: float, y0: float, h0: float, a0: float, v_max: float, h_max: float, dt):
        """
        :param x0: scalar
        :param y0: scalar
        :param h0: scalar
        :param v_max: scalar
        :param h_max: scalar
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.a = a0

        # time interval
        self.dt = dt

    def update_position(self, x_max, y_max) -> (float, float):
        """
        receive the action (heading angular rate), then update the current position
        :param y_max:
        :param x_max:
        :return:
        """
        self.a = random.uniform(-self.h_max, self.h_max)
        dx = self.dt * self.v_max * cos(self.h)  # x 方向位移
        dy = self.dt * self.v_max * sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy

        # if self.x > x_max:
        #     self.x = x_max
        # if self.x < 0:
        #     self.x = 0
        #
        # if self.y > y_max:
        #     self.y = y_max
        # if self.y < 0:
        #     self.y = 0

        # self.h += self.dt * self.a  # 更新朝向角度
        # self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内
        if 0 > self.y or self.y > y_max:
            self.h = -self.h
        elif self.x < 0 or self.x > x_max:
            if self.h > 0:
                self.h = pi-self.h
            else:
                self.h = -pi-self.h

        return self.x, self.y
