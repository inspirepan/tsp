import numpy as np
import matplotlib.pyplot as plt
from arguments import INITIALIZE_METHOD, LOCATION_RANGE, N, INPUT_LOCATIONS


# 旅行商问题建模

class TsaModel:
    # 城市数量
    N = N
    # 城市地址，(N,2)形状的numpy数组
    city_locations = None
    # 访问顺序
    visit_order = []
    ax = None

    def __init__(self, ax, seed=53):
        """
        初始化，生成城市位置，随机生成初始访问顺序
        :param ax: Matplotlib Artist对象，用于画图
        :param seed: 生成城市坐标用的种子
        """
        self.__initialize_city_locations(seed)
        self.ax = ax
        self.set_random_visit_order()

    def __initialize_city_locations(self, seed):
        """
        在范围内生成城市的坐标信息
        """
        if INITIALIZE_METHOD == "seed":
            # 根据种子，随机生成N个城市地址
            temp = np.empty((self.N, 2))
            for i in range(self.N):
                np.random.seed(i + seed)
                temp[i][0] = np.random.uniform(*LOCATION_RANGE[0])
                np.random.seed(i + 2 * seed)
                temp[i][1] = np.random.uniform(*LOCATION_RANGE[1])
            self.city_locations = temp
            np.random.seed(None)  # 取消种子
        elif INITIALIZE_METHOD == "input":
            self.city_locations = INPUT_LOCATIONS

    def draw_cities(self, ax=None, color='skyblue'):
        """
        画城市散点
        """
        if ax is None:
            ax = self.ax
        ax.set_xlim(*LOCATION_RANGE[0])
        ax.set_ylim(*LOCATION_RANGE[1])
        ax.scatter(self.city_locations[:, 0], self.city_locations[:, 1], s=30, color=color)

    def draw_path(self, ax=None, order=None, color="lightcoral"):
        """
        画路线图
        """
        if ax is None:
            ax = self.ax
        if order is None:
            order = self.visit_order[:]
        order.append(order[0])
        for i in range(1, N + 1):
            x = [self.city_locations[order[i]][0], self.city_locations[order[i - 1]][0]]
            y = [self.city_locations[order[i]][1], self.city_locations[order[i - 1]][1]]
            ax.plot(x, y, color=color)

    def set_visit_order(self, order):
        """
        设置访问顺序:
        """
        self.visit_order = order[:]

    def get_visit_order(self):
        """
        设置访问顺序:
        """
        return self.visit_order[:]

    def set_random_visit_order(self):
        """
        设置随机访问顺序
        :return:
        """
        self.visit_order = list(range(self.N))
        np.random.shuffle(self.visit_order)

    def get_path_length(self, order=None):
        """
        获取访问顺序下的路径长度
        :param order: 访问顺序
        """
        if order is None:
            order = self.visit_order[:]
        else:
            order = order[:]
        order.append(order[0])
        length = 0
        for i in range(1, N + 1):
            length += self.__distance(order[i], order[i - 1])
        return length

    def __distance(self, city_1, city_2):
        """
        计算两个城市的直线距离
        :param city_1: 城市1的序号
        :param city_2: 城市2的序号
        """
        x1, y1 = self.city_locations[city_1]
        x2, y2 = self.city_locations[city_2]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def generate_next(self):
        """
        随机生成新的访问顺序
        :return: 新的访问顺序
        """
        status = np.random.uniform(1, 3)
        if status == 1:
            return self.__switch_strategy()
        elif status == 2:
            return self.__reverse_strategy()
        else:
            return self.__insert_strategy()

    def __switch_strategy(self):
        """
        随机交换两个城市访问顺序
        """
        order = self.visit_order[:]
        i1, i2 = np.random.choice(self.N, 2, replace=False)
        temp = order[i1]
        order[i1] = order[i2]
        order[i2] = temp
        return order

    def __reverse_strategy(self):
        """
        随机选取两个城市，逆转他们之间的访问顺序:
        """
        order = self.visit_order[:]
        i1, i2 = np.random.choice(self.N, 2, replace=False)
        i1, i2 = (i1, i2) if i2 > i1 else (i2, i1)
        sub_order = order[i1:i2]
        sub_order.reverse()
        order[i1:i2] = sub_order
        return order

    def __insert_strategy(self):
        """
        随机选取两个城市，将第一个城市的访问安排到第二个城市之后
        """
        order = self.visit_order[:]
        i1, i2 = np.random.choice(self.N, 2, replace=False)
        i1, i2 = (i1, i2) if i2 > i1 else (i2, i1)
        temp = order.pop(i1)
        order.insert(i2 + 1, temp)
        return order


if __name__ == '__main__':
    ax = plt.subplot(111)
    tsa = TsaModel(ax)
    tsa.draw_cities()
    plt.show()
