import matplotlib.pyplot as plt
from matplotlib import rcParams

from arguments import EPOCHS
from function_to_solve import get_minimum_val, get_overview
from plot_utils import dot, plot_history, rgba, scatter
from sa import SimulatedAnnealing


if __name__ == '__main__':
    # 画图和打印信息设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure(dpi=70, figsize=(6, 9))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    length1 = 70
    length3 = 40

    # 模拟退火
    m = SimulatedAnnealing()
    m.print_arguments()

    # 输出待求解函数相关信息
    ax1.title.set_text("function image")
    get_overview(ax1)
    x_min, fx_min = get_minimum_val(ax1)
    print("最小点".ljust(length3, " ") + "= ({:.4f}, {:.4f})".format(x_min, fx_min))

    # 进行多次实验
    print("处理中".center(length1, "="))
    for i in range(EPOCHS):
        print(str(i + 1).ljust(2, " "), end="   ")
        x_best, fx_best, x_history, fx_history, t_history= m.solve()
        # 对第一次运行的结果画图
        if i == 0:
            dot(ax1, x_best, fx_best, annotate=True, color=rgba(255, 21, 20, 0.8), zorder=5, label='Final')
            scatter(ax1, x_history, fx_history)
            plot_history(ax2, fx_history)
            ax2.title.set_text("f(x) history")
            plot_history(ax3, x_history)
            ax3.title.set_text("x history")

    # 输出结果
    m.print_statistic_info(fx_min)
    ax1.legend()
    plt.show()
