import numpy as np


def square_grid(N, LOCATION_RANGE):
    """
    生成方格型的地图
    :return: numpy二维数组（N,2）
    """

    def get_row_col(N):
        a = np.floor(np.sqrt(N))
        while a >= 1:
            if N % a == 0:
                return int(N / a), int(a)
            a -= 1
        return int(N), 1

    array = np.empty((N, 2))
    col, row = get_row_col(N)
    for i in range(row):
        for j in range(col):
            array[i * col + j][0] = LOCATION_RANGE[0][0] + LOCATION_RANGE[0][1] / (col + 1) * (
                    j + 1) + np.random.uniform(
                *LOCATION_RANGE[0]) * 0.04
            array[i * col + j][1] = LOCATION_RANGE[1][0] + LOCATION_RANGE[1][1] / (row + 1) * (
                    i + 1) + np.random.uniform(
                *LOCATION_RANGE[1]) * 0.04
    return array


def circle(N, LOCATION_RANGE):
    """
    生成圆形地图，所有城市在一个圆上
    :param N:
    :param LOCATION_RANGE:
    :return:
    """
    len_x = (LOCATION_RANGE[0][1] - LOCATION_RANGE[0][0])
    center_x = len_x / 2 + LOCATION_RANGE[0][0]
    len_y = (LOCATION_RANGE[1][1] - LOCATION_RANGE[1][0])
    center_y = len_y / 2 + LOCATION_RANGE[1][0]
    array = np.empty((N, 2))
    size = 0.75
    for i in range(N):
        angle = np.random.uniform(0, 2 * np.pi)
        array[i][0] = center_x + len_x / 2 * np.sin(angle) * size
        array[i][1] = center_y + len_y / 2 * np.cos(angle) * size
    return array


def united_states():
    """
    需要设置参数
    N=50
    LOCATION_RANGE=((-165,-65), (15,65))
    :return: 美国各州首府坐标
    """
    print("""使用美国各州首府的坐标，需要设置参数
    N=50
    LOCATION_RANGE=((-165,-65), (15,65))""")
    return np.asarray([
        [-86.279118, 32.361538],
        [-134.41974, 58.301935],
        [-112.073844, 33.448457],
        [-92.331122, 34.736009],
        [-121.468926, 38.555605],
        [-104.984167, 39.7391667],
        [-72.677, 41.767],
        [-75.526755, 39.161921],
        [-84.27277, 30.4518],
        [-84.39, 33.76],
        [-157.826182, 21.30895],
        [-116.237651, 43.613739],
        [-89.650373, 39.78325],
        [-86.147685, 39.790942],
        [-93.620866, 41.590939],
        [-95.69, 39.04],
        [-84.86311, 38.197274],
        [-91.140229, 30.45809],
        [-69.765261, 44.323535],
        [-76.501157, 38.972945],
        [-71.0275, 42.2352],
        [-84.5467, 42.7335],
        [-93.094, 44.95],
        [-90.207, 32.32],
        [-92.189283, 38.572954],
        [-112.027031, 46.595805],
        [-96.675345, 40.809868],
        [-119.753877, 39.160949],
        [-71.549127, 43.220093],
        [-74.756138, 40.221741],
        [-105.964575, 35.667231],
        [-73.781339, 42.659829],
        [-78.638, 35.771],
        [-100.779004, 48.813343],
        [-83.000647, 39.962245],
        [-97.534994, 35.482309],
        [-123.029159, 44.931109],
        [-76.875613, 40.269789],
        [-71.422132, 41.82355],
        [-81.035, 34],
        [-100.336378, 44.367966],
        [-86.784, 36.165],
        [-97.75, 30.266667],
        [-111.892622, 40.7547],
        [-72.57194, 44.26639],
        [-77.46, 37.54],
        [-122.893077, 47.042418],
        [-81.633294, 38.349497],
        [-89.384444, 43.074722],
        [-104.802042, 41.145548]
    ])
