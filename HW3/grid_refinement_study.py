import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import forward_euler, backward_euler, crank_nicolson, exact_solution
#%%
def compute_L1_error(numerical, exact):
    return np.mean(np.abs(numerical - exact))

#%%
spatial_grid_sizes = np.array([2*1e-3*2**i for i in range(0, 10)] ) # Varying spatial grid sizes
delta_t=1e-7

fe_errors = []
be_errors = []
cn_errors = []

for delta_x in spatial_grid_sizes:
    fe_solution, grid = forward_euler(delta_x=delta_x, max_t=1e-5, delta_t=delta_t, m=2)
    be_solution, grid = backward_euler(delta_x=delta_x, max_t=1e-5, delta_t=delta_t, m=2)
    cn_solution, grid = crank_nicolson(delta_x=delta_x, max_t=1e-5, delta_t=delta_t, m=2)
    exact = exact_solution(grid, 2)

    fe_errors.append(compute_L1_error(fe_solution, exact))
    be_errors.append(compute_L1_error(be_solution, exact))
    cn_errors.append(compute_L1_error(cn_solution, exact))
    print(f"Current delta_x: {delta_x}, FE L1 Error: {fe_errors[-1]}, BE L1 Error: {be_errors[-1]}, CN L1 Error: {cn_errors[-1]}")

fe_errors = np.array(fe_errors)
be_errors = np.array(be_errors)
cn_errors = np.array(cn_errors)

#%%
plt.figure(figsize=(10,6))
plt.loglog(spatial_grid_sizes, fe_errors, label='FE Method', marker='o')
plt.loglog(spatial_grid_sizes, (spatial_grid_sizes)**2*fe_errors.mean()/spatial_grid_sizes.mean()**2, 'k--', label='O(Δx²)', alpha=0.5)
plt.xlabel('Spatial Grid Size (Δx)')
plt.ylabel('L1 Error')
plt.title('L1 Error vs Spatial Grid Size for Forward Euler Method')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.savefig('grid_refinement_plots_c_ii_a/heat_conduction_L1_error_FE.png')
plt.show()

plt.figure(figsize=(10,6))
plt.loglog(spatial_grid_sizes, be_errors, label='BE Method', marker='o')
plt.loglog(spatial_grid_sizes, (spatial_grid_sizes)**2*be_errors.mean()/spatial_grid_sizes.mean()**2, 'k--', label='O(Δx²)', alpha=0.5)
plt.xlabel('Spatial Grid Size (Δx)')
plt.ylabel('L1 Error')
plt.title('L1 Error vs Spatial Grid Size for Backward Euler Method')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.savefig('grid_refinement_plots_c_ii_a/heat_conduction_L1_error_BE.png')
plt.show()
plt.figure(figsize=(10,6))
plt.loglog(spatial_grid_sizes, cn_errors, label='CN Method', marker='o')
plt.loglog(spatial_grid_sizes, (spatial_grid_sizes)**2*cn_errors.mean()/spatial_grid_sizes.mean()**2, 'k--', label='O(Δx²)', alpha=0.5)
plt.xlabel('Spatial Grid Size (Δx)')
plt.ylabel('L1 Error')
plt.title('L1 Error vs Spatial Grid Size for Crank-Nicolson Method')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.savefig('grid_refinement_plots_c_ii_a/heat_conduction_L1_error_CN.png')
plt.show()
# %%