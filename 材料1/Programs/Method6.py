"""
描述：线性回归模型的模型评估：R方评估
作者：王佳秋
日期：2020年6月27日
"""

from sklearn import linear_model  # 表示可以调用sklearn中的linear_model模型进行线性回归
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


if __name__ == "__main__":

    X = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]

    # 1.计算样本总体平方和， y¯¯是价格 的均值， yi的训练集的第i 个价格样本，n 是样本数
    ybar = (11 + 8.5 + 15 + 18 + 11) / 5
    print("y的均值：", ybar)
    ss_tot = (11 - ybar) ** 2 + (8.5 - ybar) ** 2 + (15 - ybar) ** 2 + (18 - ybar) ** 2 + (11 - ybar) ** 2
    print("样本总体平方和：", ss_tot)

    # 2.计算残差平方和：预测值与训练样本的参测值的差异
    ss_res = (11 - 9.7759) ** 2 + (8.5 - 10.7522) ** 2 + (15 - 12.7048) ** 2 + (18 - 17.5863) ** 2 + (11 - 13.6811) ** 2
    print("残差平方和：", ss_res)

    # 3.计算R ** 2 = 1 - (ss_res/ss_tot)
    r2 = 1 - (ss_res / ss_tot)
    print("R**2: ", r2)

    # 4. 使用scikit-learn验证
    X_test = [[8], [9], [11], [16], [12]]
    y_test = [[11], [8.5], [15], [18], [11]]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    score_m = model.score(X_test, y_test)
    print("测试集model:", score_m)
