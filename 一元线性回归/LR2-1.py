"""
描述：线性回归模型-梯度下降方法1
链接：http://note.youdao.com/noteshare?id=a5654a83dd73ff80843978e6df911764&sub=197FA0D4224540E2BDB657EFA126BB7F
作者：王佳秋
日期：2020年7月2日
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def model(a, b, x):
    """模型函数"""
    return a * x + b


def cost_function(a, b, x, y):
    """损失函数"""
    n = 5  # 样本数量
    return 0.5/n * (np.square(y - a*x - b)).sum()


def optimize(a, b, x, y):
    """
    优化函数
    :param a: 标量 系数
    :param b: 标量 截距
    :param x: 向量 自变量
    :param y: 向量 因变量
    :return:
    """
    n = 5  # 样本数量
    alpha = 1e-1  # 超参数来控制迭代，通常是0.01
    y_hat = model(a, b, x)
    da = (1.0/n) * ((y_hat - y)*x).sum()
    db = (1.0/n) * ((y_hat - y).sum())
    a = a - alpha * da
    b = b - alpha * db
    return a, b


def iterate(a, b, x, y, times):
    """循环函数：多次调用优化函数来更新参数"""
    costs = []
    for i in range(times):
        print("time: ", i)
        print("运行前a: ", a, ", b: ", b)
        a, b = optimize(a, b, x, y)  # 返回新的a, b
        print("运行后a: ", a, ", b: ", b)
        costs.append(cost_function(a, b, x, y))
    return a, b, costs


if __name__ == "__main__":
    x = [13854, 12213, 11009, 10655, 9503]  # 程序员工资，顺序为北京，上海，杭州，深圳，广州
    x = np.reshape(x, newshape=(5, 1)) / 10000.0
    print("x: ", x)
    y = [21332, 20162, 19138, 18621, 18016]  # 算法工程师，顺序和上面一致
    y = np.reshape(y, newshape=(5, 1)) / 10000.0
    # 方法1：调用自己写的模型
    a = 0
    b = 0
    a, b, costs = iterate(a, b, x, y, 10000)
    print("a: ", a, ", b: ", b, "costs: ", costs)
    plt.scatter(x, y)
    y_hat = model(a, b, x)
    plt.plot(x, y_hat)
    plt.show()

    """方法2：直接采用sklearn中回归模型"""
    # lr = LinearRegression()
    # # 训练模型
    # lr.fit(x, y)
    # # 计算R平方
    # print("R平方，用于评价模型：", lr.score(x, y))
    # # 计算y_hat 预测值
    # y_hat = lr.predict(x)
    # # 打印出图
    # plt.scatter(x, y)
    # plt.plot(x, y_hat)
    # plt.show()
