"""
描述：多元线性回归模型-最小二乘法1
作者：王佳秋
日期：2020年7月11日
"""


from numpy.linalg import inv
from numpy import dot
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 全局变量
X = np.array([[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]])
y = np.array([[7], [9], [13], [17.5], [18]])


def operation_matrix():
    """通过Numpy的矩阵操作求beta矩阵"""
    beta1 = dot(inv(dot(X.T, X)), dot(X.T, y))
    print("beta1: {0}".format(beta1))
    print("type-beta1: {0}".format(type(beta1)))


def least_square_method():
    """Numpy也提供了最小二乘法函数来实现"""
    # rcond用来处理回归中的异常值，一般不用。
    beta1 = lstsq(X, y, rcond=-1)
    print("beta1: {0}".format(beta1))
    print("type-beta1: {0}".format(type(beta1)))


def least_square_method2():
    x1 = np.array([0, 1, 2, 3])
    y1 = np.array([-1, 0.2, 0.9, 2.1])
    # np.vstack()是把矩阵进行列连接。
    print(x1)
    # 增加一列值都为1的列，为系数(截距)b的x值, 如y=bx0+ax (x0=1)
    A = np.vstack([x1, np.ones(len(x1))]).T
    print("A:")
    print(A)
    print("type-A: ", type(A))
    tuple_result = np.linalg.lstsq(A, y1, rcond=-1)
    m, c = tuple_result[0]
    print("m: ", m)
    print("c: ", c)
    plt.plot(x1, y1, 'o', label='Original data', markersize=10)
    plt.plot(x1, m*x1+c, 'r', label='Fitted line')
    plt.legend()
    plt.show()


def prediction_price():
    """回归正题，有了参数，我们就来更新价格预测模型："""
    X2 = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
    y2 = [[7], [9], [13], [17.5], [18]]
    model = LinearRegression()
    model.fit(X2, y2)
    X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
    y_test = [[11], [8.5], [15], [18], [11]]
    predictions = model.predict(X_test)
    print("predictions: {0}.".format(predictions))
    for i, prediction in enumerate(predictions):
        print('Predicted: {0}, Target: {1}.'.format(prediction, y_test[i]))
    print('R-squared: {:.2f}.'.format(model.score(X_test, y_test)))


if __name__ == "__main__":
    """
        lstsq的输出包括四部分：回归系数、残差平方和、自变量X的秩、X的奇异值。一般只需要回归系数就可以了。
        Example.
        W = np.linalg.lstsq(X, Y, -1)[0]
    """
    # least_square_method()  # 方法1：最小2乘法
    # operation_matrix()  # 方法2：矩阵估算法
    # least_square_method2()  # 方法3：二元线性回归模型
    prediction_price()  # 回归正题，有了参数，我们就来更新价格预测模型：
