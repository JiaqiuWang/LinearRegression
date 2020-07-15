"""
描述：一元线性回归模型的可行性分析,将平面上的点展示在坐标系中。
链接：http://note.youdao.com/noteshare?id=35b3334eb1f550eebd7650d5873e32fe&sub=wcp1594193795636857
作者：王佳秋
日期：2020年6月19日
"""

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
    plt = runplt()
    X = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    plt.plot(X, y, 'k.')
    plt.show()
