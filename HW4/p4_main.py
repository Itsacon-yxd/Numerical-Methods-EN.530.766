import re
from elliptic_solvers import srj_jacobi
import torch
from math import sin, pi, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
delta = torch.pi / 10
N = 20
criterion = 1e-4
max_iter = 2000

w_1 = 5.0
w_2 = 0.6
w3_space = torch.linspace(0.3,2, steps=100)
converg_list = []

for w_3 in tqdm(w3_space):
    w_list = [w_1,w_2,w_3.item()]
    _, _, residual = srj_jacobi(
        u_init='xy',
        max_iter=max_iter,
        delta=delta,
        w_list=w_list
    )
    converg = torch.where(residual < criterion)[0]
    if len(converg) > 0:
        converg_list.append(converg[0].item())
    else:
        converg_list.append(max_iter)
converg_list = torch.tensor(converg_list)

converg_min = torch.min(converg_list)
w3_opt = w3_space[torch.argmin(converg_list)]
print(f'Optimal w_3: {w3_opt}, Minimum iterations to converge: {converg_min}')

_,_, residual = srj_jacobi(
    u_init='xy',
    max_iter=max_iter,
    delta=delta,
    w_list=[w_1,w_2,w3_opt]
)


plt.plot(residual)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence Behavior of SRJ Jacobi with 3 Weights')
plt.grid()
plt.savefig('p4/p4_convergence.svg')
plt.show()