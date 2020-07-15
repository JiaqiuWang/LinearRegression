"""
描述：求函数的偏导数
作者：王佳秋
日期：2020年7月7日
"""

from sympy import symbols, diff


if __name__ == "__main__":
    # 1.先对变量(x, y)符号化
    x, y = symbols('x, y', real=True)
    # 2.利用diff函数求对应函数的偏导数
    print("y的偏导数:", diff(x**2 + y**3, y))
    # 3.求出偏导数后，想求具体的值，使用subs属性进行变量的替换
    print("y的偏导数的具体值:", diff(x ** 2 + y ** 3, y).subs({x: 3, y: 1}))
    pass
