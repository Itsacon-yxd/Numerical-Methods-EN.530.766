from numerical_solvers import forward_euler, backward_euler, crank_nicolson, exact_solution
import numpy as np
import matplotlib.pyplot as plt

# Stability analysis for Forward Euler method with different r values
delta_x = 2 * np.pi / 20
max_t = 10
m = 2
r_values = [0.1, 0.25, 0.49, 0.51, 0.75, 1.0]

fig_fe, axes_fe = plt.subplots(2, 3, figsize=(15, 10))
axes_fe = axes_fe.flatten()
fig_be, axes_be = plt.subplots(2, 3, figsize=(15, 10))
axes_be = axes_be.flatten()
fig_cn, axes_cn = plt.subplots(2, 3, figsize=(15, 10))
axes_cn = axes_cn.flatten()

for idx, r in enumerate(r_values):
    fe_solution, grid = forward_euler(delta_x=delta_x, max_t=max_t, r=r, m=m)
    be_solution, grid = backward_euler(delta_x=delta_x, max_t=max_t, r=r, m=m)
    cn_solution, grid = crank_nicolson(delta_x=delta_x, max_t=max_t, r=r, m=m)

    exact = exact_solution(grid, m)
    
    # Plot at final time
    axes_fe[idx].plot(grid[0, -1], exact[0, -1], 'k-', label='Exact', linewidth=2)
    axes_fe[idx].plot(grid[0, -1], fe_solution[0, -1], 'r--', label='FE', linewidth=1.5)

    axes_be[idx].plot(grid[0, -1], exact[0, -1], 'k-', label='Exact', linewidth=2)
    axes_be[idx].plot(grid[0, -1], be_solution[0, -1], 'b--', label='BE', linewidth=1.5)

    axes_cn[idx].plot(grid[0, -1], exact[0, -1], 'k-', label='Exact', linewidth=2)
    axes_cn[idx].plot(grid[0, -1], cn_solution[0, -1], 'g--', label='CN', linewidth=1.5)

    axes_fe[idx].set_xlabel('x')
    axes_fe[idx].set_ylabel('u(x, t)')
    axes_fe[idx].set_title(f'r = {r} {"(Stable)" if r <= 0.5 else "(Unstable)"}')
    axes_fe[idx].legend()
    axes_fe[idx].grid(True, alpha=0.3)

    axes_be[idx].set_xlabel('x')
    axes_be[idx].set_ylabel('u(x, t)')
    axes_be[idx].set_title(f'r = {r}')
    axes_be[idx].legend()
    axes_be[idx].grid(True, alpha=0.3)

    axes_cn[idx].set_xlabel('x')
    axes_cn[idx].set_ylabel('u(x, t)')
    axes_cn[idx].set_title(f'r = {r}')
    axes_cn[idx].legend()
    axes_cn[idx].grid(True, alpha=0.3)

plt.tight_layout()
fig_fe.savefig('stability_analysis_plots_c_ii_b/stability_analysis_forward_euler.png')
fig_be.savefig('stability_analysis_plots_c_ii_b/stability_analysis_backward_euler.png')
fig_cn.savefig('stability_analysis_plots_c_ii_b/stability_analysis_crank_nicolson.png')

plt.show()

