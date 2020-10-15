from city_map import square_grid, circle, united_states


# 参数
EPOCHS = 10  # 实验轮次
PAUSE = 0.001  # 每次迭代暂停时间，用于图像显示
# ==========旅行商问题建模===========
N = 50  # 城市数量
# ---------地图初始化参数----------
LOCATION_RANGE = ((-165, -65), (15, 65))  # 坐标范围
# 地图初始化方法，可以选择 "random" 每次实验生成随机的地图，"seed" 按指定的种子SEED生成地图，"input" 按输入的数组生成地图
INITIALIZE_METHOD = "input"
# 如果地图初始化方法选择了"seed"，使用SEED参数指定种子
SEED = 53
# 自定义地图，当地图初始化方法选择"input"时有效，需要传入一个(N,2)的数组
# INPUT_LOCATIONS = square_grid(N, LOCATION_RANGE)
# INPUT_LOCATIONS = circle(N, LOCATION_RANGE)
INPUT_LOCATIONS = united_states()

# =========模拟退火算法===========
# --------生成初始温度参数---------
ACCEPTANCE_PROB = 0.96  # 接受概率
RANDOM_TRIAL_COUNT = 50  # 生成的随机状态总数
# ---------收敛准则参数----------
DELTA_F_LOWER_THRESHOLD = 1  # 目标函数变化下限阈值
LOW_DELTA_F_STEPS_UPPER_THRESHOLD = 10  # 连续小变化步数上限阈值
# ------------其他-----------
TEMPERATURE_DESCEND_RATE = 0.9  # 退火速率
STABLE_STEPS = 500  # 定长抽样步数
# ===========遗传算法参数============
