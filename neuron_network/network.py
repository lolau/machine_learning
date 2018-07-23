#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 神经网络反向传播算法

import random
import numpy as np
import mnist_loader


class Network(object):
    def __init__(self, sizes):
        """
        :param sizes: list类型，存储每层的神经元个数,
                     如size=[2,3,2]表示输入层、隐含层
                     和输出层的神经元个数分别为2、3、2
        """
        # 神经网络层数
        self.num_layer = len(sizes)
        self.sizes = sizes
        # 使用高斯分布随机初始化隐含层和输出层的bias
        # 隐含层(3,1) 输出层(2,1)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 使用高斯分布随机初始化隐含层到输出层，输入层到隐含层的连接的权值
        # 隐含层到输出层(3,2) 输入层到隐含层(2,3)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """
        :param a: 输入神经元(输入层或隐含层神经元)
        :return: 输出层激活值
        """
        # 第一次循环：隐含层误差、输入层到隐含层权重
        # 第二次循环：输出层误差、隐含层到输出层权重
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmod(z)

        return a

    def SGD(self, train_data, epochs, mini_batch_size, rate, test_data=None):
        """
        随机梯度下降
        :param train_data: 训练集
        :param epochs: 迭代次数
        :param mini_batch_size: 一小部分样本数量
        :param rate: 学习熟虑
        :param test_data: 测试集
        """
        train_data = list(train_data)
        n = len(train_data)
        for j in range(epochs):
            random.shuffle(train_data)

            mini_batches = [train_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_bach in mini_batches:
                self.update_mini_batch(mini_bach, rate)

            if test_data:
                test_data = list(test_data)
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, rate):
        """
        更新w 和 b
        :param mini_batch: 一小部分样本
        :param rate: 学习速率
        """
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        # 向量微分算子（倒三角）
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        # 存储每层神经元，初始只包含输入神经元
        activations = [x]

        # 存储未经sigmod计算的神经元的值
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmod(z)
            activations.append(activation)

        # 输出层delta
        delta = (activations[-1] - y) * sigmod_prime(zs[-1])

        # 输出层对b的偏导数
        nabla_b[-1] = delta
        # 输出层对w的偏导数 (l层: dot(delta(l), a(l-1).T))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 对隐含层进行反向传播
        for l in range(2, self.num_layer):
            # 从倒数第l层开始更新，使用第l+1层的delta来更新第l层的delta
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmod_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        # 通过前向传播计算出的结果
        test_result = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]

        # 返回正确识别的个数
        return sum(int(x == y) for (x, y) in test_result)


def sigmod(z):
        return 1.0 / (1.0 + np.exp(- z))


def sigmod_prime(z):
    return sigmod(z) * (1 - sigmod(z))


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
