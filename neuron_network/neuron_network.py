# 而层神经网络（输入层和输出层）

import numpy as np


def sigmod(x):
    return 1 / (1 + np.exp(- x))


# sigmod函数的导数 f'(z)=f(z)*(1-f(z))
# z = sigmod(x)
def derivative(z):
    return z * (1 - z)


def two_layer_nn(x, y):
    # 输入层与输出层之间的连接权值
    # 输入层3个神经元 * 输出层1个神经元 (3,1)
    weight = 2 * np.random.random((3, 1)) - 1

    for i in range(10000):
        # 前向传播
        input_layer = x

        # (4,3)dot(3,1)=(4,1)
        output_layer = sigmod(np.dot(input_layer, weight))

        # 反向传播
        # 误差 (4,1)
        output_layer_err = y - output_layer

        # 经确信度加权后的神经网络的误差
        # 斜率 * 误差。误差越小，修正越小（斜率小），误差越大，修正越大（斜率大） 导数范围（0,1）
        # (4,1)*(4,1)=(4,1)
        layer_1_delta = output_layer_err * derivative(output_layer)

        # 更新权重
        # (3,4)dot(4,1)=(3,1)
        weight += np.dot(input_layer.T, layer_1_delta)

    print(output_layer)


def three_layer_nn(x, y):
    weight_hidden_layer = 2 * np.random.random((3, 4)) - 1
    weight_output_layer = 2 * np.random.random((4, 1)) - 1

    for i in range(60000):
        # 正向传播
        input_layer = x
        # (4,3)dot(3,4)=(4,4)
        hidden_layer = sigmod(np.dot(input_layer, weight_hidden_layer))

        # (4,4)dot(4,1)=(4,1)
        output_layer = sigmod(np.dot(hidden_layer, weight_output_layer))

        # 反向传播
        output_layer_err = y - output_layer

        if i % 10000 == 0:
            print("Error:", str(np.mean(np.abs(output_layer_err))))

        # 输出层修正后的误差
        # (4,1)*(4,1)=(4,1)
        output_layer_delta = output_layer_err * derivative(output_layer)

        # 输出层传播的误差
        # (4,1)dot((1,4))=(4,4)
        hidden_layer_err = np.dot(output_layer_delta, weight_output_layer.T)

        # 隐藏层修正后的误差
        # (4,4)*(4,4)=(4,4)
        hidden_layer_delta = hidden_layer_err * derivative(hidden_layer)

        # (4,4)dot(4,1)=(4,1)
        weight_output_layer += np.dot(hidden_layer.T, output_layer_delta)

        # (3,4)dot(4,4)=(3,4)
        weight_hidden_layer += np.dot(input_layer.T, hidden_layer_delta)

    print(output_layer)


if __name__ == '__main__':
    x = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0, 0, 1, 1]]).T

    np.random.seed(1)

    # two_layer_nn(x, y)

    three_layer_nn(x, y)
