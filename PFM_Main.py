import numpy as np
from pyevtk.hl import imageToVTK
from microstructures import *
from laplace_operator import *
from free_energy import *

np.set_printoptions(threshold=np.inf, linewidth=300)

Nx, Ny = 160, 160
dx, dy = 0.5, 0.5
ngrain = 2

nstep = 20000
print_freq = 100
dt = 5e-3

mobil = 5.0
grcoef = 0.1

Etas = generate_2_gains(Nx, Ny)
Laplace = laplace_operator(Nx, Ny, dx, dy)

for istep in range(nstep):
    for igrain in range(ngrain):
        eta = Etas[igrain, :, :]
        DfDeta = df_deta(Etas, eta)
        LaplaceItem = Laplace.dot(eta)
        eta -= dt * mobil * (DfDeta - grcoef * LaplaceItem)
        Etas[igrain, :, :] = eta
    # print res
    if istep % print_freq == 0:
        print(f'===>step:{istep:<6} done')
        # write to file

        etas2 = np.sum(Etas ** 2, axis=0).reshape(1, Nx + 1, Ny + 1)
        imageToVTK(f"./Res/res{istep}", pointData={"data": etas2})
