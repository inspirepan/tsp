import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import cauchy

from arguments import ACCEPTANCE_PROB, DELTA_F_LOWER_THRESHOLD, EPOCHS, ETA, GENERATING_METHOD, \
    LOW_DELTA_F_STEPS_UPPER_THRESHOLD, RANDOM_TRIAL_COUNT, RANGE, SCALE, STABLE_STEPS, TEMPERATURE_DESCEND_RATE
from function_to_solve import f, get_minimum_val, get_overview
from plot_utils import dot, plot_history, rgba, scatter


class SimulatedAnnealing:
    # 用来记录统计信息
    stats_steps = []
    stats_fx_best = []
    stats_x_best = []
    stats_time = []

    @staticmethod
    def generate_init_temperature():
        """
        :return 初始温度
        """
        x = np.random.uniform(*RANGE, size=RANDOM_TRIAL_COUNT)
        best_result = np.min(f(x))
        worst_result = np.max(f(x))
        return (best_result - worst_result) / np.log(ACCEPTANCE_PROB)

    @staticmethod
    def generate_next(x):
        """
        生成新状态，使用GENERATING_METHOD设置用于状态产生的概率分布
        :param x: 当前状态
        """
        if GENERATING_METHOD == "norm":
            # 正态分布
            r = x + ETA * np.random.normal(0, SCALE)
            while r > RANGE[1] or r < RANGE[0]:
                r = x + ETA * np.random.normal(0, SCALE)
            return r
        elif GENERATING_METHOD == "cauchy":
            # 柯西分布
            r = x + ETA * cauchy.rvs(loc=0, scale=SCALE, size=1)
            while r > RANGE[1] or r < RANGE[0]:
                r = x + ETA * cauchy.rvs(loc=0, scale=SCALE, size=1)
            return r
        else:
            # 均匀分布
            return np.random.uniform(*RANGE)

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

    def solve(self):
        """
        采用模拟退火算法求解最小值
        """
        start_time = time.time()
        fx_history = []  # 记录目标函数值变化
        x_history = []
        t_history = []  # 记录温度变化
        steps = 0
        # 生成初始温度
        init_temperature = self.generate_init_temperature()
        t = init_temperature
        # 随机生成初始解
        x = np.random.uniform(*RANGE)
        fx = f(x)
        fx_prev = fx
        low_delta_f_steps = 0
        # 模拟退火算法
        # 收敛准则
        while low_delta_f_steps < LOW_DELTA_F_STEPS_UPPER_THRESHOLD:
            # 记录步数
            steps += 1
            # 定长步抽样
            for i in range(STABLE_STEPS):
                # 产生新状态
                x_next = float(self.generate_next(x))
                fx_next = f(x_next)
                # 接受函数判断
                if self.acceptance_probability(fx - fx_next, t) > np.random.uniform(0, 1):
                    x, fx = x_next, fx_next
            # 记录每次满足抽样稳定准则后的信息
            fx_history.append(fx)
            x_history.append(x)
            t_history.append(t)
            # 退火降温
            t *= TEMPERATURE_DESCEND_RATE
            # 计算Δf，用于判断收敛准则
            delta_f = abs(fx - fx_prev)
            fx_prev = fx
            if delta_f < DELTA_F_LOWER_THRESHOLD:
                low_delta_f_steps += 1
            else:
                low_delta_f_steps = 0
        fx_best = np.min(fx_history)
        x_best = x_history[int(np.argmin(fx_history))]
        end_time = time.time()
        self.stats_x_best.append(x_best)
        self.stats_fx_best.append(fx_best)
        self.stats_steps.append(steps)
        self.stats_time.append(end_time - start_time)
        print("({:.4f}, {:.4f})   初始温度 = {:.4g}  迭代次数 = {}  耗时 = {:.2f}s".format(x_best, fx_best, init_temperature,
                                                                                 steps,
                                                                                 end_time - start_time))
        return x_best, fx_best, x_history, fx_history, t_history, steps, init_temperature

    def get_statistic_info(self, fx_min):
        length1 = 70
        length2 = 10
        print()

        def mse(stats: list, target: int):
            """均方误差"""
            return np.mean(list(map(lambda x: (x - target) ** 2, stats)))

        print("统计信息".center(length1, "="))
        print("目标函数最小值".center(length1, '-'))
        print("平均值".ljust(length2, " ") + "= {:.4f}".format(np.mean(self.stats_fx_best)))
        print("方差".ljust(length2, " ") + "= {:.2e}".format(np.var(self.stats_fx_best)))
        print("均方误差".ljust(length2, " ") + "= {:.2e}".format(mse(self.stats_fx_best, fx_min)))
        print("迭代次数".center(length1, '-'))
        print("平均值".ljust(length2, " ") + "= {:.2f}".format(np.mean(self.stats_steps)))
        print("耗时".center(length1, '-'))
        print("平均值".ljust(length2, " ") + "= {:.2f}s".format(np.mean(self.stats_time)))
        print("方差".ljust(length2, " ") + "= {:.2e}".format(np.var(self.stats_time)))

    @staticmethod
    def get_arguments():
        # 打印参数
        print("参数".center(length1, "="))
        print("初始温度接受系数".ljust(length3, " ") + "= {:.2f}".format(ACCEPTANCE_PROB))
        print("确定初始温度时生成的随机状态总数".ljust(length3, " ") +
              "= {:.2f}".format(RANDOM_TRIAL_COUNT))
        print("退火温度下降速率".ljust(length3, " ") + "= {:.2f}".format(TEMPERATURE_DESCEND_RATE))
        print("新状态的生成方法".ljust(length3, " ") + "= " + GENERATING_METHOD.upper())
        if GENERATING_METHOD == "cauchy":
            print("尺度系数".ljust(length3, " ") + "= {:.2f}".format(SCALE))
            print("扰动幅度参数".ljust(length3, " ") + "= {:.2f}".format(ETA))
        elif GENERATING_METHOD == "gauss":
            print("标准差".ljust(length3, " ") + "= {:.2f}".format(SCALE))
            print("扰动幅度参数".ljust(length3, " ") + "= {:.2f}".format(ETA))
        print("抽样定长步数".ljust(length3, " ") + "= {}".format(STABLE_STEPS))
        print("目标值变化下限阈值".ljust(length3, " ") + "= {:.4f}".format(DELTA_F_LOWER_THRESHOLD))
        print("连续目标值变化小于下限步数上限阈值".ljust(length3, " ") + "= {}".format(LOW_DELTA_F_STEPS_UPPER_THRESHOLD))


if __name__ == '__main__':
    # 画图和打印信息设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure(dpi=600, figsize=(6, 9))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    length1 = 70
    length3 = 40

    sa = SimulatedAnnealing()
    sa.get_arguments()

    # 输出待求解函数相关信息
    ax1.title.set_text("function image")
    get_overview(ax1)
    x_min, fx_min = get_minimum_val(ax1)
    print("最小点".ljust(length3, " ") + "= ({:.4f}, {:.4f})".format(x_min, fx_min))

    # 进行多次实验
    print("处理中".center(length1, "="))
    for i in range(EPOCHS):
        print(str(i + 1).ljust(2, " "), end="   ")
        x_best, fx_best, x_history, fx_history, t_history, steps, init_temperature = sa.solve()
        # 对第一次运行的结果画图
        if i == 0:
            dot(ax1, x_best, fx_best, annotate=True, color=rgba(255, 21, 20, 0.8), zorder=5, label='Final')
            scatter(ax1, x_history, fx_history)
            plot_history(ax2, fx_history)
            ax2.title.set_text("f(x) history")
            plot_history(ax3, x_history)
            ax3.title.set_text("x history")

    # 输出结果
    sa.get_statistic_info(fx_min)
    ax1.legend()
    plt.show()
