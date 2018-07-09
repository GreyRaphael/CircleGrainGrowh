import numpy as np
from scipy import sparse, linalg


def laplace_operator(Nx, Ny, dx, dy):
    nx = Nx + 1
    ny = Ny + 1
    factor = 1 / (dx * dy)

    r = np.zeros(ny)
    r[:2] = [-4 * factor, 1 * factor]
    T = linalg.toeplitz(r)
    E = sparse.eye(nx)

    laplace = sparse.kron(E, T, format='coo')
    print(laplace.getformat())
    laplace.setdiag(factor, k=ny)
    laplace.setdiag(factor, k=-ny)
    laplace.setdiag(factor, k=nx * ny - ny)
    laplace.setdiag(factor, k=-nx * ny + ny)

    laplace = laplace.tolil()
    for j in range(nx):
        ii = j * ny
        jj = (j + 1) * ny - 1
        laplace[ii, jj] = factor
        laplace[jj, ii] = factor

    return laplace


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=300)
    Laplace = laplace_operator(4, 3, 0.5, 0.5)

    print(Laplace.toarray())
    print(Laplace.shape)
