import numpy as np

from arguments import GENE_LENGTH, K, N, PROB_CROSSOVER, PROB_MUTATION, RANGE
from function_to_solve import f


class GeneticAlgorithms:
    # 记录历史结果
    stats_fx_best = []
    stats_x_best = []

    def solve(self):
        """
        使用遗传算法求解一次
        :return:
        """
        x_history = []  # 每次种群迭代中最佳的结果
        fx_history = []  # 每次迭代最佳结果对应的函数值

        p = self.get_initial_generation()  # 种群P
        # 记录初始种群中的最佳结果
        curr_best = self.get_curr_best(p)
        x_history.append(curr_best)
        fx_history.append(f(curr_best))
        next_p = []  # 下一代
        f_v = self.get_fitness_value(p)  # 适配值
        for k in range(K):
            m = 0
            while m < N:
                temp1, temp2 = self.select_two(p, f_v)
                if PROB_CROSSOVER > np.random.uniform(0, 1):
                    temp1, temp2 = self.crossover(temp1, temp2)
                temp1, temp2 = self.mutation(temp1, temp2)
                m += 2
                next_p.append(temp1)
                next_p.append(temp2)
            p = next_p
            next_p = []
            f_v = self.get_fitness_value(p)
            curr_best = self.get_curr_best(p)
            x_history.append(curr_best)
            fx_history.append(f(curr_best))
        x_best = x_history[int(np.argmin(fx_history))]
        fx_best = np.min(fx_history)
        self.stats_x_best.append(x_best)
        self.stats_fx_best.append(fx_best)
        print("({:.4f}, {:.4f})".format(x_best, fx_best))
        return x_best, fx_best, x_history, fx_history

    def print_statistic_info(self, fx_min):
        """
        输出多次运行后的统计信息
        """
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

    @staticmethod
    def encoding(de):
        """
        编码
        :param de: 一组十进制的变量值
        :return: 转换成一组二进制表示的基因序列数组
        """

        def simple_encoding(d):
            temp = (d - RANGE[0]) / (RANGE[1] - RANGE[0]) * (2 ** GENE_LENGTH - 1)
            temp = np.floor(temp)
            gene_array = []
            for i in range(GENE_LENGTH):
                current_power = GENE_LENGTH - i - 1
                current_divide = np.power(2, current_power)
                gene_array.append(int(np.floor(temp / current_divide)))
                temp %= current_divide
            return gene_array

        return np.asarray(list(map(simple_encoding, de)))

    @staticmethod
    def decoding(bi):
        """
        解码
        :param bi: 一组二进制的基因序列数组
        :return: 转换成一组十进制的变量值
        """

        def simple_decoding(b):
            i = GENE_LENGTH - 1
            temp = 0
            for digit in b:
                temp += digit * 2 ** i
                i -= 1
            return temp / (2 ** GENE_LENGTH - 1) * (RANGE[1] - RANGE[0]) + RANGE[0]

        return np.asarray(list(map(simple_decoding, bi)))

    @staticmethod
    def get_fitness_value(p):
        f_v = f(GeneticAlgorithms.decoding(p))
        max_f = max(f_v)
        return (max_f - f_v + 1) / (max_f - min(f_v) + 1)

    @staticmethod
    def get_initial_generation():
        """
        随机获取初始化种群
        :return:N个随机的个体
        """
        return np.random.randint(low=0, high=2, size=(N, GENE_LENGTH)).astype(np.int8)

    @staticmethod
    def select_two(p, fitness_value):
        """
        根据适配值选择个体
        :param p: 当前代
        :param fitness_value:适配值
        :return:选出两个个体的索引
        """
        summ = sum(fitness_value)
        idx1, idx2 = np.random.choice(np.arange(N), replace=True, size=2, p=fitness_value / summ)
        return p[idx1], p[idx2]

    @staticmethod
    def crossover(temp1, temp2):
        """
        交叉操作
        """
        i1, i2 = np.random.choice(np.arange(GENE_LENGTH), size=2, replace=True)
        if i1 > i2:
            i1, i2 = i2, i1
        i2 += 1
        c1 = np.append(temp1[:i1], temp2[i1:(i2 + 1)])
        c1 = np.append(c1, temp1[i2:])
        c2 = np.append(temp2[:i1], temp1[i1:(i2 + 1)])
        c2 = np.append(c2, temp2[i2:])
        return c1, c2

    @staticmethod
    def mutation(temp1, temp2):
        """
        变异操作
        """
        for i in range(GENE_LENGTH):
            if PROB_MUTATION > np.random.uniform(0, 1):
                if temp1[i] == 0:
                    temp1[i] = 1
                else:
                    temp1[i] = 0
            if PROB_MUTATION > np.random.uniform(0, 1):
                if temp2[i] == 0:
                    temp2[i] = 1
                else:
                    temp2[i] = 0
        return temp1, temp2

    @staticmethod
    def get_curr_best(p):
        """
        返回当前种群中的最优解
        :param p:
        :return:
        """
        fv = GeneticAlgorithms.get_fitness_value(p)
        idx = np.argmax(fv)
        binary_result = p[idx]
        return GeneticAlgorithms.decoding([binary_result])[0]


if __name__ == '__main__':
    ga = GeneticAlgorithms()
    for i in range(10):
        ga.solve()
    ga.print_statistic_info()
