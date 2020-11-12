import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot, dot, rgba
from arguments import RANGE, PRECISION
from matplotlib import rcParams


def f(x):
    """定义用于求解的多极小值函数"""
    return (5 * np.sin(5 * x) + 4 * np.cos(6 * x)) * (np.exp(-np.absolute(x / 10 - 0.5)))


def get_minimum_val(ax):
    """
    遍历计算最小值
    :return 范围内最小值的坐标(x,y)
    """
    min_y = 20
    min_x = 0
    for x in np.arange(*RANGE, PRECISION):
        if f(x) <= min_y:
            min_y = f(x)
            min_x = x
    dot(ax, min_x, min_y, annotate=False, color=rgba(67, 160, 71, 0.7), zorder=5, s=18, label="Least")
    return min_x, min_y


def get_overview(ax):
    """函数概览，画出函数图像，标出最小点"""
    x = np.arange(*RANGE, PRECISION)
    plot(ax, x, f(x), lw=1)


# 输出待求解函数预览
if __name__ == '__main__':
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    get_overview(ax)
    get_minimum_val(ax)
    plt.legend()
    plt.show()
