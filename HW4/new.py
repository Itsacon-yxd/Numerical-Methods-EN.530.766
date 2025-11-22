import torch
import matplotlib.pyplot as plt
from elliptic_solvers import sor_jacobi, sor_gauss_seidel
from joblib import Parallel, delayed
from math import cos, sqrt, pi

_,_,residual = sor_gauss_seidel(u_init='xy', max_iter=1000, delta=torch.pi/10, w=1.3)
print(f"Grid Shape: {_.shape},Solution Shape: {_.shape}")
plt.plot(residual)
plt.yscale('log')
plt.grid()
plt.show()