import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


def trunc_gauss_random(loc, scale, left, right, size):
    """
    :return: 返回size个(left, right)内的正态分布的随机数
    """
    cl = norm.cdf(left, loc=loc, scale=scale)
    cr = norm.cdf(right, loc=loc, scale=scale)
    return norm.ppf(np.random.uniform(cl, cr, size=size), loc=loc, scale=scale)


def trunc_cauchy_random(loc, scale, left, right, size):
    """
    :return: 返回size个(left, right)内的柯西分布的随机数
    """
    cl = np.arctan((left - loc) / scale) / np.pi + 0.5
    cr = np.arctan((right - loc) / scale) / np.pi + 0.5
    return loc + scale * np.tan(np.pi * (-0.5 + np.random.uniform(cl, cr, size)))


# plt.subplot(211)
scale = 4
size = 10000000
# t1 = time.time()
# plt.hist(trunc_gauss_random(5, sigma, -10, 10, size), bins=10000)
# plt.title("normal")
t2 = time.time()
# print(t2 - t1)
# plt.subplot(212)
plt.hist(trunc_cauchy_random(5, scale, -10, 10, size), bins=10000)
plt.title("cauchy")
t3 = time.time()
print(t3 - t2)
plt.show()
