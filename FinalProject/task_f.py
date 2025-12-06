import jax.numpy as jnp
from jax_solver import Pt_GS_Solver
import jax
from time import time
import matplotlib.pyplot as plt

grid_sizes = [32, 96, 160, 224]
max_iter = 15000

boundary_cond = jnp.array([
        [0.5, 0.5, 0.25],
        [0.25, 0.25, 0.125],
        [0.25, 0.75, 0.125],
        [0.75, 0.25, 0.125],
        [0.75, 0.75, 0.125]
    ])

solvers_prev = [Pt_GS_Solver(num_pts=grid_size, boundary_list=boundary_cond[:1]) for grid_size in grid_sizes]
solvers_new = [Pt_GS_Solver(num_pts=grid_size, boundary_list=boundary_cond[1:]) for grid_size in grid_sizes]

cpu_times_prev = []
cpu_times_new = []
res_prev = []
res_new = []

for solver in solvers_prev:
    _ = solver.solve(max_iter=2)  # Warm-up
    start_time = time()
    solution, residual = solver.solve(max_iter=max_iter)
    solution.block_until_ready()
    end_time = time()
    cpu_times_prev.append(end_time - start_time)
    converged = jnp.argmax(residual < 1e-4)
    res_prev.append(converged)

for solver in solvers_new:
    _ = solver.solve(max_iter=2)  # Warm-up
    start_time = time()
    solution, residual = solver.solve(max_iter=max_iter)
    solution.block_until_ready()
    end_time = time()
    cpu_times_new.append(end_time - start_time)
    converged = jnp.argmax(residual < 1e-4)
    res_new.append(converged)

converge_times_prev = [cpu_times_prev[i]*res_prev[i]/max_iter for i in range(len(res_prev))]
converge_times_new = [cpu_times_new[i]*res_new[i]/max_iter for i in range(len(res_new))]

plt.plot(grid_sizes, cpu_times_prev, label='Previous Boundary Conditions', marker='o')
plt.plot(grid_sizes, cpu_times_new, label='New Boundary Conditions', marker='o')
plt.plot(grid_sizes, jnp.array(cpu_times_prev[0])*(jnp.array(grid_sizes)/jnp.array(grid_sizes[0]))**2, '--', label='O(N^2) Reference')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Grid Size (N)')
plt.ylabel('CPU Time (s)')
plt.grid()
plt.title('CPU Time for Fixed Iterations')
plt.legend()
plt.savefig('task_f/cpu_time_fixed_iterations.svg')
plt.show()

plt.plot(grid_sizes, res_prev, label='Previous Boundary Conditions', marker='o')
plt.plot(grid_sizes, res_new, label='New Boundary Conditions', marker='o')
plt.plot(grid_sizes, jnp.array(res_prev[0])*(jnp.array(grid_sizes)/jnp.array(grid_sizes[0]))**2, '--', label='O(N^2) Reference')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Grid Size (N)')
plt.ylabel('Iterations to Convergence')
plt.grid()
plt.title('Number of Iterations to Convergence')
plt.legend()
plt.savefig('task_f/convergence_iterations.svg')
plt.show()

plt.plot(grid_sizes, converge_times_prev, label='Previous Boundary Conditions', marker='o')
plt.plot(grid_sizes, converge_times_new, label='New Boundary Conditions', marker='o')
plt.plot(grid_sizes, jnp.array(converge_times_prev[0])*(jnp.array(grid_sizes)/jnp.array(grid_sizes[0]))**4, '--', label='O(N^4) Reference')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Grid Size (N)')
plt.ylabel('Time to Convergence (s)')
plt.grid()
plt.title('Time to Convergence')
plt.legend()
plt.savefig('task_f/convergence_time.svg')
plt.show()

ratio = jnp.array(res_new) / jnp.array(res_prev)
plt.plot(grid_sizes, ratio, marker='o')
plt.xlabel('Grid Size (N)')
plt.ylabel('Ratio of Iterations (New / Previous)')
plt.grid()
plt.title('Ratio of Iterations to Convergence')
plt.savefig('task_f/iteration_ratio.svg')
plt.show()