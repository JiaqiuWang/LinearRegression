"""
描述：多项式回归
链接：http://note.youdao.com/noteshare?id=7856797175ba7ccbc1e8e96e57a4a52c&sub=109606BFD7E24CF68C919BF51E81985F
作者：王佳秋
日期：2020年7月13日
"""


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=10)
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签


def runplt(size=None):
    plt.figure(figsize=size)
    plt.title("披萨价格与直径数据", fontproperties=font)
    plt.xlabel('直径（英寸）', fontproperties=font)
    plt.ylabel('价格（美元）', fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt


def polynomial_regression_quadratic():
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    x_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    xx = np.linspace(0, 26, 100)  # [0~26分100份]
    # print(xx)
    # print(xx.shape[0])  # shape函数返回list，array,matrix等的一维和二维长度值。
    # print(xx.reshape(xx.shape[0], 1))  # shape函数返回list，array,matrix等的一维和二维长度值。
    yy = regressor.predict(xx.reshape(xx.shape[0], 1))
    plt = runplt(size=(8, 8))
    plt.plot(X_train, y_train, 'k.', label="train")
    plt.plot(xx, yy, label='一元线性回归')

    # 多项式回归
    # degree 表示多项式的维度，即^2， interaction_only表示是否仅使用a*b,include_bias表示是否引入偏执项1
    quadratic_feature = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)  # degree：控制多项式的度
    X_train_quadratic = quadratic_feature.fit_transform(X_train)  # 利用均值与方差对训练集进行数据标准化
    X_test_quadratic = quadratic_feature.transform(x_test)
    regressor_quadratic = LinearRegression()

    # 训练数据集用来fit拟合
    regressor_quadratic.fit(X_train_quadratic, y_train)
    xx_quadratic = quadratic_feature.transform(xx.reshape(xx.shape[0], 1))
    print("xx_quadratic")
    print(xx_quadratic)

    # 测试数据集用来predict预测
    plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-', label='多项式回归')
    plt.legend()
    plt.show()
    print()
    print("X_train:")
    print(X_train)
    print("X_train_quadratic:")
    print(X_train_quadratic)
    print("X_test:")
    print(x_test)
    print("X_test_quadratic:")
    print(X_test_quadratic)
    print('一元线性回归r-squared: ', regressor.score(x_test, y_test))
    print('二次回归r-squared: ', regressor_quadratic.score(X_test_quadratic, y_test))


def polynomial_regression_cubic():
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    x_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    xx = np.linspace(0, 26, 100)  # [0~26分100份]

    plt = runplt()
    plt.plot(X_train, y_train, 'k.')
    quadratic_feature = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_train_quadratic = quadratic_feature.fit_transform(X_train)
    X_test_quadratic = quadratic_feature.transform(x_test)
    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic, y_train)
    xx_quadratic = quadratic_feature.transform(xx.reshape(xx.shape[0], 1))
    plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-', label='quadratic regression')

    cubic_feature = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
    X_train_cubic = cubic_feature.fit_transform(X_train)
    X_test_cubic = cubic_feature.transform(x_test)
    regressor_cubic = LinearRegression()
    regressor_cubic.fit(X_train_cubic, y_train)
    xx_cubic = cubic_feature.transform(xx.reshape(xx.shape[0], 1))

    plt.plot(xx, regressor_cubic.predict(xx_cubic), label="cubic regression")
    plt.legend()
    plt.show()

    print("X_train_cubic:")
    print(X_train_cubic)
    print("X_test_cubic:")
    print(X_test_cubic)

    print("2次回归 r-squared: ", regressor_quadratic.score(X_test_quadratic, y_test))
    print("3次回归 r-squared: ", regressor_cubic.score(X_test_cubic, y_test))


def polynomial_regression_7th():
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    x_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    xx = np.linspace(0, 26, 100)  # [0~26分100份]

    plt = runplt()
    plt.plot(X_train, y_train, 'k.', label="训练数据点")
    quadratic_feature = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_train_quadratic = quadratic_feature.fit_transform(X_train)
    X_test_quadratic = quadratic_feature.transform(x_test)
    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic, y_train)
    xx_quadratic = quadratic_feature.transform(xx.reshape(xx.shape[0], 1))
    plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-', label='2次多项式回归')

    seventh_feature = PolynomialFeatures(degree=7, interaction_only=False, include_bias=True)
    X_train_seventh = seventh_feature.fit_transform(X_train)
    X_test_seventh = seventh_feature.transform(x_test)
    regressor_seventh = LinearRegression()
    regressor_seventh.fit(X_train_seventh, y_train)
    xx_seventh = seventh_feature.transform(xx.reshape(xx.shape[0], 1))
    plt.plot(xx, regressor_seventh.predict(xx_seventh), label='7次多项式回归')
    plt.show()

    print("2次回归 r-squared: ", regressor_quadratic.score(X_test_quadratic, y_test))
    print("7次回归 r-squared: ", regressor_seventh.score(X_test_seventh, y_test))


if __name__ == "__main__":
    # polynomial_regression_quadratic()  # 2次方多项式回归
    # polynomial_regression_cubic()  # 3次方多项式回归
    polynomial_regression_7th()  # 7次方多项式回归
