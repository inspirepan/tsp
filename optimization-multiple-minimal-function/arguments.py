# 参数
EPOCHS = 20  # 实验轮次
RANGE = (-5, 5)  # 求解范围
PRECISION = 0.0001  # 遍历求解最小解以及画图的精度

# =========模拟退火算法===========
# -------生成初始温度的参数--------
ACCEPTANCE_PROB = 0.9  # 接受概率
RANDOM_TRIAL_COUNT = 50  # 生成的随机状态总数
# --------状态发生器参数----------
GENERATING_METHOD = "cauchy"  # 新状态产生方法，"uniform": 均匀分布，"norm": 正态分布，"cauchy": 柯西分布，默认使用均匀分布
ETA = (RANGE[1] - RANGE[0]) * 0.1  # 扰动幅度参数
SCALE = 1  # 正态分布的sigma值，柯西分布的尺度参数
# ---------收敛准则参数-----------
DELTA_F_LOWER_THRESHOLD = PRECISION  # 目标函数变化下限阈值
LOW_DELTA_F_STEPS_UPPER_THRESHOLD = 10  # 连续小变化步数上限阈值
# -----------其他参数------------
TEMPERATURE_DESCEND_RATE = 0.9  # 退火速率
STABLE_STEPS = 100  # 定长抽样步数
# ==========遗传算法============


