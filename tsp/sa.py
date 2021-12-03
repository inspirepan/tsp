import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from arguments import ACCEPTANCE_PROB, DELTA_F_LOWER_THRESHOLD, EPOCHS, INITIALIZE_METHOD, \
    LOW_DELTA_F_STEPS_UPPER_THRESHOLD, PAUSE, RANDOM_TRIAL_COUNT, SEED, STABLE_STEPS, TEMPERATURE_DESCEND_RATE
from tsp_model import TsaModel


class SimulatedAnnealing:
    # 用于生成地图的种子
    seed = None
    epoch = 0
    best_order = None
    best_result = None

    def __init__(self):
        if INITIALIZE_METHOD == "seed":
            self.seed = SEED

    @staticmethod
    def generate_init_temperature(tsa):
        """
        :return 获取初始温度
        """
        c = []
        for i in range(RANDOM_TRIAL_COUNT):
            tsa.set_random_visit_order()
            c.append(tsa.get_path_length())
        return (min(c) - max(c)) / np.log(ACCEPTANCE_PROB)

    @staticmethod
    def acceptance_probability(delta_val, temp):
        """
        接受概率函数，使用Metropolis准则
        :param delta_val: 旧值减去新值
        :param temp: 当前温度
        :return: 接受概率，位于[0,1]内
        """
        if delta_val > 0:
            return 1
        else:
            return np.exp(delta_val / temp)

    def solve(self, ax1, ax2, ax3, ax4, ax5):
        """
        使用模拟退火算法求解TSP问题
        :param ax1: 画出地图与当前路径
        :param ax2: 画出当前轮次路径迭代变化
        :param ax3: 画出目前的最佳结果
        :param ax4: 画出当前轮次温度迭代变化
        :param ax5: 画出上一轮的结果
        :return:
        """
        ax1.cla()
        ax2.cla()
        ax4.cla()
        if self.epoch == 0:
            ax3.set_title("最短路径长度 = NaN")
            ax5.set_title("上一轮路径长度 = NaN")
        self.epoch += 1
        path_length_history = []  # 记录目标函数值变化
        visit_order_history = []
        t_history = []  # 记录温度变化
        steps = 0
        tsa = TsaModel(ax1, self.seed)
        # 生成初始温度
        init_temperature = self.generate_init_temperature(tsa)
        t = init_temperature

        path_length = tsa.get_path_length()
        prev_path_length = path_length
        low_delta_f_steps = 0
        # 模拟退火算法
        # 收敛准则
        while low_delta_f_steps < LOW_DELTA_F_STEPS_UPPER_THRESHOLD:
            # 记录步数
            steps += 1
            # 定长步抽样
            for i in range(STABLE_STEPS):
                # 产生新状态
                new_order = tsa.generate_next()
                new_path_length = tsa.get_path_length(new_order)
                # 接受函数判断
                if self.acceptance_probability(path_length - new_path_length, t) > np.random.uniform(0, 1):
                    tsa.set_visit_order(new_order)
                    path_length = new_path_length
            # 画出本轮退火前的图像
            plt.pause(PAUSE)
            ax1.cla()
            ax1.set_title(
                "第{}轮   初始温度 = {:.0f}   迭代次数 = {}".format(self.epoch, init_temperature, steps))
            tsa.draw_cities()
            tsa.draw_path()
            # 记录信息
            path_length_history.append(path_length)
            visit_order_history.append(tsa.get_visit_order())
            t_history.append(t)
            # 画图
            ax2.plot(np.arange(0, steps, 1), path_length_history, lw=1, color='goldenrod')
            ax2.set_title("当前路径长度 = {:.2f}".format(path_length))
            ax4.plot(np.arange(0, steps, 1), t_history, lw=1, color='orangered')
            ax4.set_title("当前温度 = {:.2f}".format(t))
            # 退火降温
            t *= TEMPERATURE_DESCEND_RATE
            # 计算Δf，用于判断收敛准则
            delta_f = abs(path_length - prev_path_length)
            prev_path_length = path_length
            if delta_f < DELTA_F_LOWER_THRESHOLD:
                low_delta_f_steps += 1
            else:
                low_delta_f_steps = 0
        # 记录当前轮次的最佳结果
        path_length_best = np.min(path_length_history)
        visit_order_best = visit_order_history[int(np.argmin(path_length_history))]
        # 记录所有轮次最佳结果
        if self.epoch == 1:
            self.best_order = visit_order_best
            self.best_result = path_length_best
        else:
            if path_length_best <= self.best_result:
                self.best_order = visit_order_best
                self.best_result = path_length_best
        ax3.cla()
        ax3.set_title("最短路径长度 = {:.2f}".format(self.best_result))
        tsa.draw_cities(ax3, color="turquoise")
        tsa.draw_path(ax3, self.best_order, color="teal")

        ax5.cla()
        ax5.set_title("上一轮路径长度 = {:.2f}".format(path_length_best))
        tsa.draw_cities(ax5)
        tsa.draw_path(ax5, visit_order_best)
        return path_length_best, path_length_history, t_history, steps, init_temperature


if __name__ == '__main__':
    # 画图和打印信息设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['SIMHEI']
    rcParams['axes.unicode_minus'] = False
    mpl.use("Qt5Agg")
    plt.ion()  # 打开交互模式
    fig = plt.figure(dpi=70, figsize=(15, 7.5))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 4), (0, 2))
    ax3 = plt.subplot2grid((2, 4), (1, 2))
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax5 = plt.subplot2grid((2, 4), (1, 3))
    length1 = 70
    length2 = 10
    length3 = 40

    # 打印参数
    print("参数".center(length1, "="))
    print("初始温度接受系数".ljust(length3, " ") + "= {:.2f}".format(ACCEPTANCE_PROB))
    print("确定初始温度时生成的随机状态总数".ljust(length3, " ") + "= {:.0f}".format(RANDOM_TRIAL_COUNT))
    print("退火温度下降速率".ljust(length3, " ") + "= {:.2f}".format(TEMPERATURE_DESCEND_RATE))
    print("抽样定长步数".ljust(length3, " ") + "= {}".format(STABLE_STEPS))
    print("目标值变化下限阈值".ljust(length3, " ") + "= {:.4f}".format(DELTA_F_LOWER_THRESHOLD))
    print("连续小变化步数上限阈值".ljust(length3, " ") + "= {}".format(LOW_DELTA_F_STEPS_UPPER_THRESHOLD))

    # 用来记录统计信息
    stats_steps = []
    stats_path_length_best = []
    stats_time = []

    # 进行多次实验
    sa = SimulatedAnnealing()
    if INITIALIZE_METHOD == "random":
        print("当前种子".ljust(length3, " ") + "= {}".format(sa.seed))
    print("处理中".center(length1, "="))
    for i in range(EPOCHS):
        print(str(i + 1).ljust(2, " "), end="   ")
        start_time = time.time()
        path_length_best, path_length_history, t_history, steps, init_temperature = sa.solve(ax1, ax2, ax3, ax4, ax5)
        end_time = time.time()
        # 输出结果
        print("路径长度 = {:.2f}  初始温度 = {:.4g}  迭代次数 = {}  耗时 = {:.2f}s".format(path_length_best,
                                                                             init_temperature,
                                                                             steps,
                                                                             end_time - start_time))
        stats_path_length_best.append(path_length_best)
        stats_steps.append(steps)
        stats_time.append(end_time - start_time)

    # 输出统计信息
    print()
    print("统计信息".center(length1, "="))
    print("目标函数最小值".center(length1, '-'))
    print("平均值".ljust(length2, " ") + "= {:.4f}".format(np.mean(stats_path_length_best)))
    print("方差".ljust(length2, " ") + "= {:.2e}".format(np.var(stats_path_length_best)))
    print("迭代次数".center(length1, '-'))
    print("平均值".ljust(length2, " ") + "= {:.2f}".format(np.mean(stats_steps)))
    print("耗时".center(length1, '-'))
    print("平均值".ljust(length2, " ") + "= {:.2f}s".format(np.mean(stats_time)))
    print("方差".ljust(length2, " ") + "= {:.2e}".format(np.var(stats_time)))
    # ax.legend()
    plt.ioff()
    plt.show()
