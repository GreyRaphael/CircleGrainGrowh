import numpy as np
import pylab as plt


def generate_2_gains(nx, ny):
    x, y = np.ogrid[:nx+1, :ny+1]

    radius = nx / 3
    etas = np.zeros((2, nx+1, ny+1))
    etas[0, :, :] = 1
    etas[0, :, :][(x - nx / 2) ** 2 + (y - ny / 2) ** 2 < radius ** 2] = 0.0
    etas[1, :, :][(x - nx / 2) ** 2 + (y - ny / 2) ** 2 < radius ** 2] = 1.0

    return etas.reshape(2, -1, 1)


if __name__ == '__main__':
    Etas = generate_2_gains(80, 80)
    print(Etas.shape)
    # # imshow results
    # for i in range(2):
    #     fig = plt.figure()
    #     plt.imshow(Etas[i, :, :])
    # plt.show()
