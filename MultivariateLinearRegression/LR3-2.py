"""
描述：多元线性回归模型-最小二乘法1
作者：王佳秋
日期：2020年7月14日
"""


import numpy as np
import pandas as pd
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot  # 矩阵点乘
from numpy import mat  # 二维矩阵


def linear_regression_one():
    X = mat([1, 2, 3])  # 将队列转换成矩阵
    print("X: ", X, ", type: ", type(X))
    X = X.reshape(3, 1)  # X为1,2,3
    print("reshape X: ", X)
    Y = mat([5, 10, 15]).reshape(3, 1)
    a = dot(inv(dot(X.T, X)), dot(X.T, Y))  # a为系数/斜率
    print("斜率a: ")
    print(a)


def multivariate_linear_regression():
    dataset = pd.read_csv('data2.csv')  # 读入数据
    print("dataset:")
    print(dataset)
    X = dataset.iloc[:, 2:5]  # X为所有行，2到4列
    print("X:")
    print(X)
    Y = dataset.iloc[:, 1]  # Y为所有行，第1列
    print("Y:")
    print(Y)
    a = dot(dot(inv(np.dot(X.T, X)), X.T), Y)  # 最小二乘法求解公式
    print("系数a: ")
    print(a)


if __name__ == "__main__":
    # linear_regression_one()
    multivariate_linear_regression()
