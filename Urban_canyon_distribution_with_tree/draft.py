import numpy as np
from sympy import *
import random
x = np.exp(-0.527 * 4.9)         # 真实路径长度计算的
x1 = np.exp(-0.328 * 4.9)        # 球形理论计算公式
# # y = np.exp(-0.61 * 4.9)
# print('椭球透射率的模拟值为{}'.format(x))
# print('圆球透射率的模拟值为{}'.format(x1))
# y = np.mean([0.096, 0.113, 0.129, 0.100, 0.104, 0.116, 0.117, 0.144, 0.143])
# print('透射率的实测值为{}'.format(y))
# z = np.mean([0.063, 0.038, 0.032, 0.030, 0.043, 0.048, 0.043, 0.085, 0.1])
# print('同济的模拟值为{}'.format(z))
# # # print(y)
# # # x = symbols('x')
# # # a = 0.47 + 1.27 * np.sin(x)
# # # b = 1.1 * np.cos(x) / (1 + 0.2 * a)
# # # f = np.sqrt(a ** 2 + b ** 2)
# # # A = integrate(f, (x, 0, 2 * np.pi))
# # print(A)
# # for i in np.arange(0, 2):
# #   i = random.uniform(63, 70)
# #   print(i)
print(np.exp(-0.48 * 4.91))
