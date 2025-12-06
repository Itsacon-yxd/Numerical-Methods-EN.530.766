import jax.numpy as jnp
from jax_solver import Pt_GS_Solver, harmonic_solution, get_accuracy
import jax
from time import time
import matplotlib.pyplot as plt

grid_sizes = [32, 96, 160, 224]
boundary_condition = jnp.array([[0.5, 0.5, 0.25]])
max_iter = 10000
num_steps = 10

# %% task e I Convergence vs Relaxation Factor
w_ls = jnp.linspace(1.0, 2.0, num_steps)
start_time = time()
solvers = [Pt_GS_Solver(grid_size,boundary_condition) for grid_size in grid_sizes]

res_ls = []
for solver in solvers:
    def solve_wrapped(w):
        solution, residual = solver.solve(w=w, max_iter=max_iter)
        conv_mask = residual < 1e-4
        has_converged = jnp.any(conv_mask)
        converge_idx = jnp.argmax(conv_mask)
        return jax.lax.select(has_converged, converge_idx, max_iter)

    converge_idxs = jax.vmap(solve_wrapped)(w_ls)
    res_ls.append(list(converge_idxs))
end_time = time()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

for i, grid_size in enumerate(grid_sizes):
    row = i // 2
    col = i % 2
    ax[row, col].semilogy(w_ls, res_ls[i], marker='o')
    ax[row, col].set_title(f'Grid Size: {grid_size}x{grid_size}')
    ax[row, col].set_xlabel('Relaxation Factor (w)')
    ax[row, col].set_ylabel('Iterations to Converge')
    ax[row, col].grid()

plt.tight_layout()
plt.savefig('task_e/convergence_vs_w.svg')
plt.show()

# %% task e II CPU Time vs Grid Size
# grid_sizes=[32,96,160,224,288,352,416,512]
solvers = [Pt_GS_Solver(grid_size,boundary_condition) for grid_size in grid_sizes]
cpu_time = []
converged = []
for solver, grid_size in zip(solvers, grid_sizes):
    _ = solver.solve(max_iter=2)
    start_time = time()
    solution, residual = solver.solve(w=1.8, max_iter=max_iter)
    solution.block_until_ready()
    end_time = time()
    converged.append(jnp.argmax(residual < 1e-4))
    cpu_time.append(end_time - start_time)

plt.plot(grid_sizes, cpu_time, marker='o')
plt.plot(grid_sizes, jnp.array(cpu_time[0])*(jnp.array(grid_sizes)/jnp.array(grid_sizes[0]))**2, '--', label='O(N^2) Reference')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Grid Size')
plt.ylabel('CPU Time (seconds)')
plt.title('CPU Time for 10000 Iterations vs Grid Size')
plt.grid()
plt.legend()
plt.savefig('task_e/cpu_time_vs_grid_size.svg')
plt.show()

cpu_time_converged = [cpu_time[i] * (converged[i]/max_iter) for i in range(len(grid_sizes))]
plt.plot(grid_sizes, cpu_time_converged, marker='o')
plt.plot(grid_sizes, jnp.array(cpu_time_converged[0])*(jnp.array(grid_sizes)/jnp.array(grid_sizes[0]))**4, '--', label='O(N^4) Reference')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Grid Size')
plt.ylabel('CPU Time to Converge (seconds)')
plt.title('CPU Time to Converge vs Grid Size')
plt.grid()
plt.legend()
plt.savefig('task_e/cpu_time_to_converge_vs_grid_size.svg')
plt.show()

# %% task e III Accuracy vs Grid Size
grid_sizes = [32, 96, 160, 224]
solvers = [Pt_GS_Solver(grid_size,boundary_condition) for grid_size in grid_sizes]
u_exact_ls = [harmonic_solution(
    solver.grid[1],solver.grid[0],center=boundary_condition[0,:-1],radius = boundary_condition[0,-1]
    ).at[solver.inner_boundary].set(1.0) 
              for solver in solvers]

err_ls = []
for idx, solver in enumerate(solvers):
    u_numerical, residual = solver.solve(outer_bc=u_exact_ls[idx][solver.whole_boundary],w=1.5, max_iter=max_iter)
    accuracy = get_accuracy(u_numerical, u_exact_ls[idx])
    err_ls.append(accuracy)

plt.plot(grid_sizes, err_ls, marker='o')
plt.plot(grid_sizes, jnp.array(err_ls[0])*(jnp.array(grid_sizes[0])/jnp.array(grid_sizes))**2, '--', label='O(h^2) Reference')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Grid Size')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Grid Size')
plt.legend()
plt.grid()
plt.savefig('task_e/error_vs_grid_size.svg')
plt.show()