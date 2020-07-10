"""
描述：线性回归模型-梯度下降方法2
链接：http://note.youdao.com/noteshare?id=467c5a80f9205edad6ac09ac02b85d4a&sub=49C6D1030BCF4051932C5BD2C75FD34D
作者：王佳秋
日期：2020年7月9日
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def linear_regression():
    learning_rate = 0.01  # 学习步长
    initial_b = 0  # 截距
    initial_m = 0  # 斜率
    num_iter = 1000  # 迭代次数

    data = init_data()  # 初始化数据
    print("data:")
    print(data)
    print("type-data: ", type(data))
    [b, m] = optimizer_two(data, initial_b, initial_m, learning_rate, num_iter)
    plot_data(data, b, m)
    print("b: {0}, m: {1}.".format(b, m))


def plot_data(data, b, m):
    x = data[:, 0]
    y = data[:, 1]
    y_predict = m * x + b
    plt.plot(x, y, 'o')
    plt.plot(x, y_predict, 'k-')
    plt.show()


def init_data():
    data = np.loadtxt('data.csv', delimiter=',')
    return data


def optimizer(data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 100 == 0:
            print("times: {0}, errors: {1}".format(i, compute_error(b, m, data)))
    return [b, m]


def compute_gradient(b_cur, m_cur, data, learning_rate):
    """每次迭代计算梯度，做参数更新"""
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))  # 样本数据的个数

    # 偏导数，梯度
    for i in range(0, len(data)):  # 循环每个数据样例
        x = data[i, 0]
        y = data[i, 1]

        b_gradient += -(2/N) * (y - (m_cur * x + b_cur))  # 偏导数
        m_gradient += -(2/N) * x * (y - (m_cur * x + b_cur))

    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)
    return [new_b, new_m]


def compute_error(b, m, data):
    totalError = 0
    x = data[:, 0]  # 在二维数组中，取所有行的第1个元素
    # print("x: ", x)
    y = data[:, 1]  # 在二维数组中，取所有行的第2个元素
    # print("y: ", y)
    totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)
    return totalError / len(data)


def optimizer_two(data, initial_b, initial_m, learning_rate, num_iter):
    """公式非常难以计算的情况下怎么去求最优解，此时求偏导数可以使用导数的定义"""
    b = initial_b
    m = initial_m

    while True:
        before = compute_error(b, m, data)
        print("before: {0}".format(before))
        b, m = compute_gradient_two(b, m, data, learning_rate)
        after = compute_error(b, m, data)
        print("after: {0}".format(after))
        if abs(after - before) < 0.000000001:  # 不断减小精度
            break
    return [b, m]


def compute_gradient_two(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    delta = 0.0000001

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        # 利用导数的定义来计算梯度（自查百度百科）
        b_gradient += (error(x, y, b_cur + delta, m_cur) - error(x, y, b_cur - delta, m_cur)) / (2 * delta)
        m_gradient += (error(x, y, b_cur, m_cur + delta) - error(x, y, b_cur, m_cur - delta)) / (2 * delta)

    b_gradient = b_gradient / N
    m_gradient = m_gradient / N

    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)
    return [new_b, new_m]


def error(x, y, b, m):
    """损失函数"""
    return (y - (m * x) - b) ** 2


def scikit_learn():
    data = init_data()
    print("data:")
    print(data)
    y = data[:, 1]
    x = data[:, 0]
    print("y:")
    print(y)
    print("x:")
    print(x)
    x = x.reshape(-1, 1)
    print("x':")
    print(x)
    linreg = LinearRegression()
    linreg.fit(x, y)
    print("斜率：{}".format(linreg.coef_))
    print("截距：{}".format(linreg.intercept_))


if __name__ == "__main__":
    linear_regression()  # 自己动手写的一元线性回归，已知误差函数，且其偏导数较难求得的情况。
    """sklearn中有相应的方法求线性回归，其直接使用最小二乘法求最优解。简单实现以做个比较。"""
    scikit_learn()
