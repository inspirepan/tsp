import numpy as np


# 一些绘图实用函数

def plot(ax, x, y, lw=1):
    """画x-y折线"""
    ax.plot(x, y, lw=lw, color='steelblue')
    ax.grid()


def scatter(ax, x, y, s=10, color=None):
    """画x-y散点"""
    if color is None:
        color = rgba(255, 167, 38, 0.6)
    ax.scatter(x, y, s=s, color=color, zorder=1)


def rgba(r, g, b, a):
    return r / 255, g / 255, b / 255, a


def plot_history(ax, y, lw=1):
    """画变化曲线"""
    ax.plot(np.arange(0, len(y), 1), y, lw=lw, color='steelblue')
    ax.grid()


def dot(ax, x, y, annotate=False, color='orange', zorder=4, label=None, s=10):
    """画单个点"""
    ax.scatter([x], [y], s=s, color=color, zorder=zorder, label=label)
    if annotate:
        ax.annotate("({:.4f}, {:.4f})".format(x, y), (x + 0.5, y + 0.5))
