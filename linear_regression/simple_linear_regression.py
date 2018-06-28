import numpy as np
import matplotlib.pyplot as plt
import timeit


# 一元线性回归，模型：y=m*x+b
def linear_aggression():
    data = np.loadtxt("data.csv", delimiter=',')

    np.random.shuffle(data)  # make data random

    init_m = 0
    init_b = 0
    learn_rate = 0.01
    delta = 1e-5

    train_set_size = int(len(data) * (2 / 3))

    train_set = data[: train_set_size]
    test_set = data[train_set_size:]

    print("---------------batch gradient descent----------------")

    start_time_bgd = timeit.default_timer()
    m_bgd, b_bgd = batch_gradient_descent(train_set, init_m, init_b, learn_rate, delta)
    end_time_bgd = timeit.default_timer()

    print("Train time %.6f:" % (end_time_bgd - start_time_bgd))

    print("Train set error: ", compute_error(train_set, m_bgd, b_bgd))
    print("Test set error: ", compute_error(test_set, m_bgd, b_bgd))

    plot_data(train_set, test_set, m_bgd, b_bgd)

    print("--------------stochastic gradient descent---------------")

    start_time_sgd = timeit.default_timer()
    m_sgd, b_sgd = stochastic_gradient_descent(train_set, init_m, init_b, learn_rate, delta)
    end_time_sgd = timeit.default_timer()

    print("Train time %.6f:" % (end_time_sgd - start_time_sgd))

    print("Train set error: ", compute_error(train_set, m_sgd, b_sgd))
    print("Test set error: ", compute_error(test_set, m_sgd, b_sgd))

    plot_data(train_set, test_set, m_sgd, b_sgd)


def batch_gradient_descent(data, init_m, init_b, learn_rate, delta):
    m_grad = init_m
    b_grad = init_b
    iterations = 0

    while True:
        new_m, new_b = one_bgd(data, m_grad, b_grad, learn_rate)

        if np.sum(abs(new_m - m_grad)) < delta:
            print("converged.")
            break

        if iterations % 100 == 0:
            err = compute_error(data, m_grad, b_grad)
            print("Iteration: %d -- Error: %.6f" % (iterations, err))

        iterations += 1

        m_grad = new_m
        b_grad = new_b

    return [m_grad, b_grad]


def one_bgd(data, m, b, learn_rate):
    ogd_m = 0
    ogd_b = 0
    n = len(data)

    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]

        ogd_m += (2 / n) * (y - (m * x + b)) * (-x)
        ogd_b += (2 / n) * (y - (m * x + b)) * (-1)

    new_m = m - learn_rate * ogd_m
    new_b = b - learn_rate * ogd_b

    return [new_m, new_b]


def stochastic_gradient_descent(data, init_m, init_b, learn_rate, delta):
    m_grad = init_m
    b_grad = init_b
    iterations = 0

    while True:
        new_m, new_b = one_sgd(data, m_grad, b_grad, learn_rate)

        if np.sum(abs(new_m - m_grad)) < delta:
            print("Converged.")
            break

        if iterations % 100 == 0:
            err = compute_error(data, m_grad, b_grad)
            print("Iteration: %d -- Error: %.6f" % (iterations, err))

        iterations += 1

        m_grad = new_m
        b_grad = new_b

    return [m_grad, b_grad]


def one_sgd(data, m, b, learn_rate):
    sgd_m = m
    sgd_b = b
    n = len(data)

    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]

        sgd_m = sgd_m - learn_rate * (2 / n) * (y - (m * x + b)) * (-x)
        sgd_b = sgd_b - learn_rate * (2 / n) * (y - (m * x + b)) * (-1)

    return [sgd_m, sgd_b]


def normal_equation():
    return


def compute_error(data, m, b):
    x = data[:, 0]
    y = data[:, 1]
    error = (y - (m * x + b)) ** 2
    total_error = np.sum(error, axis=0)

    return total_error / len(data)


def plot_data(train_set, test_set, m, b):
    x_train = train_set[:, 0]
    y_train = train_set[:, 1]

    y_predict = m * x_train + b

    x_test = test_set[:, 0]
    y_test = test_set[:, 1]

    plt.plot(x_train, y_train, 'o')

    plt.plot(x_test, y_test, 'ro')

    plt.plot(x_train, y_predict, 'k')

    plt.show()


if __name__ == '__main__':
    linear_aggression()

