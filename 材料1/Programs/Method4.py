"""
描述：判断哪一条回归直线为最佳拟合的
作者：王佳秋
日期：2020年6月21日
"""

from sklearn import linear_model  # 表示可以调用sklearn中的linear_model模型进行线性回归
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


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
    model.fit(X, y)
    print(model.intercept_)  # 截距
    print(model.coef_)  # 线性模型的系数
    a = model.predict([[12]])
    print("预测一张12英寸匹萨的价格: {:.2f}".format(model.predict([[12]])[0][0]))

    # plt = runplt()
    # plt.plot(X, y, 'k.')
    X2 = [[0], [10], [14], [25]]
    # X2 = np.array(X2)
    print("X2: ", X2)
    model = linear_model.LinearRegression()
    model.fit(X, y)
    y2 = model.predict(X2)
    print("y2: ", y2)
    # y2 = np.array(y2).reshape(1, 4)
    print("type-y2: ", type(y2))
    print(y2)

    plt = runplt(size=(10, 10))
    # plt.plot(X, y, 'k.')
    y3 = [14.25, 14.25, 14.25, 14.25]
    # y3 = np.array(y3).reshape(1, 4)
    y4 = y2 * 0.5 + 5
    # y4 = np.array(y4).reshape(1, 4)
    model.fit(X[1:-1], y[1:-1])
    y5 = model.predict(X2)
    # y5 = np.array(y5).reshape(1, 4)
    plt.plot(X, y, 'k.', label='X,y')
    plt.plot(X2, y2, 'g-.', label='X2,y2')
    plt.plot(X2, y3, 'r-.', label="X2,y3")
    plt.plot(X2, y4, 'y-.', label="X2,y4")
    plt.plot(X2, y5, 'o-.', label="X2,y5")
    plt.legend()
    plt.show()
