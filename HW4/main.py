import torch
import matplotlib.pyplot as plt
from jacobi import jacobi,get_residual

torch.manual_seed(0)  # For reproducibility

delta=torch.pi/10
init_list=['zero','xy','random']
residual_list = []

fig_solution, axes_solution = plt.subplots(1, len(init_list), figsize=(15, 5))
fig_residual, axes_residual = plt.subplots(1, len(init_list), figsize=(15, 5))
for init, ax_solution, ax_residual in zip(init_list, axes_solution, axes_residual):
    print(f'Initial condition: {init}')
    
    grid, solution, residual = jacobi(u_init=init, delta=delta)
    residual_list.append(residual)

    res_plot = get_residual(solution, delta, sum=False)

    im_residual = ax_residual.imshow(res_plot.squeeze())
    ax_residual.set_title(f'{init} Initial Condition Residual')
    ax_residual.set_xlabel('x')
    ax_residual.set_ylabel('y')
    ax_residual.invert_yaxis()  # Invert y-axis to match the typical Cartesian coordinate system

    im_original = ax_solution.imshow(solution.squeeze())
    ax_solution.set_title(f'{init} Initial Condition Solution')
    ax_solution.set_xlabel('x')
    ax_solution.set_ylabel('y')
    ax_solution.invert_yaxis()  # Invert y-axis to match the typical Cartesian coordinate system

fig_solution.colorbar(im_original, ax=axes_solution, orientation='vertical')
fig_residual.colorbar(im_residual, ax=axes_residual, orientation='vertical')
fig_solution.suptitle('Jacobi Method Solutions for Different Initial Conditions')
fig_residual.suptitle('Jacobi Method Residuals for Different Initial Conditions')
fig_solution.savefig('jacobi_solutions.png', dpi=300)
fig_residual.savefig('jacobi_residuals.png', dpi=300)
plt.close(fig_solution)
plt.close(fig_residual)

plt.figure(figsize=(10, 6))
for init, residual in zip(init_list, residual_list):
    plt.plot(residual, label=f'{init} Initial Condition')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual (log scale)')
plt.title('Residual Convergence of Jacobi Method for Different Initial Conditions')
plt.legend()
plt.grid()
plt.savefig('jacobi_residual_convergence.png', dpi=300)
plt.close()