import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 使用scikit-learn库的LinearRegression方法
def linear_aggression():
    data = np.loadtxt("Advertising.csv", delimiter=',')
    x_data = data[:, 1:4]
    y_data = data[:, 4:]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1)

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    a = linreg.coef_
    b = linreg.intercept_

    m = np.array([np.array([b[0], a[0][0], a[0][1], a[0][2]])]).T

    print(m)

    x_test = np.insert(x_test, 0, values=np.ones([len(x_test)]), axis=1)

    predict_curve(x_test, y_test, m)


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