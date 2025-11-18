import torch
import matplotlib.pyplot as plt
from elliptic_solvers import jacobi,get_residual, gauss_seidel

torch.manual_seed(42)  # For reproducibility

delta=torch.pi/10
N = 10
criterion=1e-3
max_iter=2000
init_list=['xy','random']
jacobi_residual_list = []
gs_residual_list = []

for init in init_list:
    print(f'Initial condition: {init}')
    
    grid, solution, residual = jacobi(u_init=init, delta=delta, max_iter=max_iter)
    jacobi_residual_list.append(residual)

    grid, solution, residual = gauss_seidel(u_init=init, delta=delta, max_iter=max_iter)
    gs_residual_list.append(residual)

plt.figure(figsize=(10, 6))
for init, jacobi_residual, gs_residual in zip(init_list, jacobi_residual_list, gs_residual_list):
    iters = torch.arange(len(jacobi_residual))
    plt.plot(iters, jacobi_residual, label=f'Jacobi Residual')
    plt.plot(iters, gs_residual, label=f'Gauss-Seidel Residual')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log scale)')
    plt.title('Residual Convergence of Jacobi vs Gauss-Seidel')
    plt.legend()
    plt.grid()
    plt.savefig('p1/'+init+'_convergence.svg')
    plt.close()

    jacobi_curr = jacobi_residual[0]
    gs_curr = gs_residual[0]
    jacobi_rate = [0]
    gs_rate = [0]

    for iter in range(max_iter):
        jacobi_rate[-1]+=1
        jacobi_next = jacobi_residual[iter]
        if jacobi_curr/jacobi_next>10:
            jacobi_curr=jacobi_next
            jacobi_rate.append(0)


        if jacobi_residual[iter] < criterion:
            print(f'Jacobi converged for initial condition "{init}" at iteration {iter}')
            break
    for iter in range(max_iter):
        gs_rate[-1]+=1
        gs_next = gs_residual[iter]
        if gs_curr/gs_next>10:
            gs_curr=gs_next
            gs_rate.append(0)


        if gs_residual[iter] < criterion:
            print(f'Gauss-Seidel converged for initial condition "{init}" at iteration {iter}')
            break

    jacobi_rate = torch.tensor(jacobi_rate)
    gs_rate = torch.tensor(gs_rate)
    print(f'Jacobi convergence rate: {jacobi_rate.float().mean():.2f} iterations per 10^{-1}')
    print(f'Gauss-Seidel convergence rate: {gs_rate.float().mean():.2f} iterations per 10^{-1}')
