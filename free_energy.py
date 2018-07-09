import numpy as np


def df_deta(etas, eta):
    A = 1.0
    B = 1.0
    others_sum = np.sum(etas ** 2, axis=0) - eta ** 2
    return A * (2.0 * B * eta * others_sum + eta ** 3 - eta)


if __name__ == '__main__':
    ngrain = 2
    arr = np.zeros((ngrain, 4, 4))
    arr[0, :, :] = np.arange(1, 17).reshape(4, 4)
    arr[1, :, :] = np.arange(101, 117).reshape(4, 4)

    etas = arr.reshape(ngrain, -1, 1)