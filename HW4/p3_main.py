from matplotlib.pylab import f
from elliptic_solvers import srj_jacobi
import torch
from math import sin, pi, sqrt
import matplotlib.pyplot as plt

torch.manual_seed(42)
delta = torch.pi / 10
N = 20
criterion = 1e-3
max_iter = 2000

w_list = [3.414213] + 2*[0.585786]

grid, solution, residual = srj_jacobi(
    u_init='xy',
    max_iter=max_iter,
    delta=delta,
    w_list=w_list
)
converg = torch.where(residual < criterion)[0]

print("Converged in", converg[0].item(), "iterations.")

plt.plot(residual)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence Behavior of SRJ Jacobi with 2 Weights')
plt.grid()
plt.savefig('p3/p3_convergence.svg')
plt.show()