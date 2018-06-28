import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn import preprocessing


#  多元线性回归，模型：y = m0 + m1*x1 + m2*x2 + m3*x3
def linear_aggression():
    # 去掉第一列的编号
    data_raw = np.loadtxt("Advertising.csv", delimiter=',')[:, 1:]

    np.random.shuffle(data_raw)  # make data random

    # 特征缩放
    data = preprocessing.scale(data_raw)

    data = np.insert(data, 0, values=1, axis=1)

    train_set_size = int(len(data) * (2 / 3))

    train_set = data[: train_set_size]
    test_set = data[train_set_size:]

    x_train = train_set[:, :4]
    y_train = train_set[:, 4:]

    x_test = test_set[:, :4]
    y_test = test_set[:, 4:]

    print("--------------normal equation---------------")

    start_time_bgd = timeit.default_timer()
    m_vector = normal_equation(x_train, y_train)
    end_time_bgd = timeit.default_timer()

    print("Parameter:", m_vector)

    print("Train time %.6f:" % (end_time_bgd - start_time_bgd))
    print("Train set error: ", compute_error(x_train, y_train, m_vector))
    print("Test set error: ", compute_error(x_test, y_test, m_vector))

    predict_curve(x_test, y_test, m_vector)


def normal_equation(x_train, y_train):

    xTx = np.dot(x_train.T, x_train)
    if np.linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse.")
        return
    m_vector = np.dot(np.linalg.inv(xTx), (np.dot(x_train.T, y_train)))

    return m_vector


def compute_error(x, y, m_vector):
    error = (y - np.dot(x, m_vector.reshape(-1, 1))) ** 2

    total_error = np.sum(error, axis=0)

    return total_error / len(x)


def predict_curve(x, y, m_vector):
    y_predict = np.dot(x, m_vector)

    t = np.arange(len(x))

    plt.plot(t, y, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_predict, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    linear_aggression()

