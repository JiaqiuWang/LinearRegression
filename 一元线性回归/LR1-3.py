"""
描述：采用scikit-learn构建一元线性回归模型，并绘制模型直线
链接：http://note.youdao.com/noteshare?id=35b3334eb1f550eebd7650d5873e32fe&sub=wcp1594193795636857
作者：王佳秋
日期：2020年6月20日
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

    plt = runplt()
    plt.plot(X, y, 'k.')

    X2 = [[0], [10], [14], [25]]
    print("X2: ", X2, " type: ", type(X2))
    X2 = np.array(X2)  # 这个加不加都可以
    print("X2: ", X2, " type: ", type(X2))

    y2 = model.predict(X2)
    print("y2: ", y2)
    print("type-y2: ", type(y2))
    print("y2': ", y2)

    plt.plot(X2, y2, 'g-.', label='X2,y2')
    plt.show()
