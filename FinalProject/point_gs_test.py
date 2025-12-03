from concurrent.futures import ProcessPoolExecutor
import re
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_solver import LaplaceSolver
from time import time

grid_sizes = [32, 96, 160, 224]
boundary_condition = jnp.array([[0.5, 0.5, 0.25]])

def solve_grid(N, repeat=10):
    solver = LaplaceSolver(num_pts=N, boundary_list=boundary_condition)
    solution, residuals = solver.solve(max_iter=2, w=1.5)
    start_time = time()
    for _ in range(repeat):
        solution, residuals = solver.solve(max_iter=12000, w=1.5)
        solution.block_until_ready()
    end_time = time()
    return solution, residuals, end_time - start_time

# Run solvers in parallel
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(solve_grid, grid_sizes))

    solution_list, residual_list, time_list = zip(*results)

    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    for ax, solution, N in zip(axs.flatten(), solution_list, grid_sizes):
        im = ax.imshow(solution, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
        ax.set_title(f'Grid Size: {N}x{N}')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('p_c/laplace_solutions.svg')
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 6))
    for residuals, N in zip(residual_list, grid_sizes):
        ax.semilogy(residuals, label=f'Grid Size: {N}x{N}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence of Residuals')
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('p_c/laplace_residuals.svg')
    plt.show()
    print("Execution times for each grid size:")
    for N, t in zip(grid_sizes, time_list):
        print(f"Grid size {N}x{N}: {t:.2f} seconds")