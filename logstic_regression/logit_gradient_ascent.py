import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import train_test_split


# 使用梯度上升，模型为 y = sigmod(w*x)
def logistic_regression():
    learn_rate = 0.001
    times = 500

    x, y = init_data()

    # 留出法划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print("---------------batch gradient descent----------------")

    start_time_bgd = timeit.default_timer()
    weights = batch_gradient_ascent(x_train, y_train, learn_rate, times)
    end_time_bgd = timeit.default_timer()

    print("Train time %.6f:" % (end_time_bgd - start_time_bgd))

    print("Train set accuracy %.4f %%: " % (compute_accuracy(weights, x_train, y_train) * 100))
    print("Test set accuracy %.4f %%: " % (compute_accuracy(weights, x_test, y_test) * 100))

    weights = np.mat(weights).transpose()
    print(weights)

    plot_pic(weights)

    print("---------------stochastic gradient descent----------------")

    start_time_bgd = timeit.default_timer()
    weights = stoc_grad_ascent(x_train, y_train, learn_rate, times)
    end_time_bgd = timeit.default_timer()

    print("Train time %.6f:" % (end_time_bgd - start_time_bgd))

    print("Train set accuracy %.4f %%: " % (compute_accuracy(weights, x_train, y_train) * 100))
    print("Test set accuracy %.4f %%: " % (compute_accuracy(weights, x_test, y_test) * 100))

    weights = np.mat(weights).transpose()
    print(weights)

    plot_pic(weights)


def init_data():
    data = np.loadtxt('logistic_data.csv')

    x = data[:, : 2]
    # ad x_0
    x = np.insert(x, 0, values=1, axis=1)
    y = data[:, -1]
    return x, y


def batch_gradient_ascent(x, y, learn_rate, times):
    # 需要转成矩阵，否则直接使用*会报错
    x_matrix = np.mat(x)

    y_mat_t = np.mat(y).transpose()

    # train_size, parameter_size
    m, n = np.shape(x_matrix)

    weights = np.ones((n, 1))

    for i in range(times):
        h = sigmoid(x_matrix * weights)
        weights = weights + learn_rate * x_matrix.transpose() * (y_mat_t - h)

    return weights


# 随机梯度上升，由于存在嵌套循环，耗时更长
def stoc_grad_ascent(x, y, learn_rate, times):
    x_matrix = np.mat(x)
    y_mat_t = np.mat(y).transpose()

    m, n = np.shape(x_matrix)

    weights = np.ones((n, 1))

    for k in range(times):
        for i in range(m):
            h = sigmoid(x_matrix[i, :] * weights)
            weights = weights + learn_rate * x_matrix[i, :].transpose() * (y_mat_t[i, :] - h)

    return weights


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_accuracy(weights, x, y):
    y = np.mat(y).T
    m = len(x)
    match_count = 0

    for i in range(m):
        predict = sigmoid(x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(y[i, 0]):
            match_count += 1

    accuracy = float(match_count) / m

    return accuracy


def plot_pic(weights):
    x, y = init_data()
    n = np.shape(x)[0]
    x1, x2, y1, y2 = [], [], [], []

    for i in range(n):
        if y[i] == 1:
            x1.append(x[i][1])
            y1.append(x[i][2])
        else:
            x2.append(x[i][1])
            y2.append(x[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)

    print(weights)

    y_pred = (-weights[0, 0] - weights[0, 1] * x) / weights[0, 2]

    ax.plot(x, y_pred)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    logistic_regression()