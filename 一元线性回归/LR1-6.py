"""
描述：解一元线性回归的最小二乘法
链接：http://note.youdao.com/noteshare?id=35b3334eb1f550eebd7650d5873e32fe&sub=wcp1594193795636857
作者：王佳秋
日期：2020年6月21日
"""

from sklearn import linear_model  # 表示可以调用sklearn中的linear_model模型进行线性回归
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


font = FontProperties(fname=r'C:\windows\fonts\msyh.ttc', size=10)


def runplt(size=None):
    plt.figure(figsize=size)
    plt.title('匹萨价格与直径距离', fontproperties=font)
    plt.xlabel('直径（英寸）', fontproperties=font)
    plt.ylabel('价格（美元）', fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt


if __name__ == "__main__":

    model = linear_model.LinearRegression()
    X = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]

    # x 的均值:xbar
    xbar = (6 + 8 + 10 + 14 + 18) / 5
    print("x的均值：", xbar)

    # 方差：variance
    variance = ((6 - xbar) ** 2 + (8 - xbar) ** 2 + (10 - xbar) ** 2 + (14 - xbar) ** 2 + (18 - xbar) ** 2) / (5 - 1)
    print("方差：", variance)

    """
    Numpy里面有var方法可以直接计算方差，ddof参数是：Delta Degrees of Freedom“：计算中使用的除数是”N-ddof“，
    其中”N“代表元素的数量。默认情况下，”ddof“为零。设置为1，可得样本方差无偏估计量
    """
    # import numpy as np
    print("采用np包计算的方差：", np.var(X, ddof=1))

    # 协方差的计算
    ybar = (7 + 9 + 13 + 17.5 + 18) / 5
    print("y的均值：", ybar)
    cov = ((6 - xbar) * (7 - ybar) + (8 - xbar) * (9 - ybar) + (10 - xbar) * (13 - ybar) + (14 - xbar) * (17.5 - ybar) + (18 - xbar) * (18 - ybar)) / (5 - 1)
    print("x,y的协方差: ", cov)
    """Numpy里面有cov方法可以直接计算协方差"""
    print("np中的协方差计算方法：", np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1])

    # 计算β系数（斜率）：cov(X, y) / var(X)
    β = cov / variance
    print("β系数：", β)

    # 截距α = ybar - β * xbar
    α = ybar - β * xbar
    print("截距α: ", α)
