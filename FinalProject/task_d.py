import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_solver import Pt_GS_Solver, harmonic_solution
from time import time
import jax
# jax.config.update("jax_enable_x64", True)

grid_size = 224
boundary_condition = jnp.array([[0.5, 0.5, 0.25]])
w = 1.5
max_iter = 20000

def line_exact_solution(r):
    # r can be a scalar or array
    cond1 = (r > 0.25) & (r < 0.75)
    cond2 = (r >= 0) & (r <= 0.25)
    cond3 = (r >= 0.75) & (r < 1)
    # Assign value for each region
    y = jnp.where(cond1, 1,
        jnp.where(cond2, 4 * r,
        jnp.where(cond3, 4 - 4 * r, 0)))  # Default 0 for r outside [0,1]
    return y

    

solver = Pt_GS_Solver(num_pts = grid_size, boundary_list=boundary_condition)

solution_line,residual = solver.solve(w=w,max_iter=max_iter)
exact_line = line_exact_solution(solver.x)

plt.semilogy(residual)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History with Original Boundary Condition')
plt.grid()
plt.savefig('task_d/convergence_line.svg')
plt.show()

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].plot(solution_line[grid_size//2])
axes[0,0].plot(exact_line,'--')
axes[0,0].set_title('Solution Line at Middle Grid')
axes[0,0].grid()

axes[0,1].plot(jnp.abs(solution_line[grid_size//2] - exact_line))
axes[0,1].set_title('Absolute Error along Middle Grid Line')
axes[0,1].grid()

axes[1,0].plot(solution_line[:,grid_size//2])
axes[1,0].plot(exact_line,'--')
axes[1,0].set_title('Solution Line at Middle Grid')
axes[1,0].grid()

axes[1,1].plot(jnp.abs(solution_line[:,grid_size//2] - exact_line))
axes[1,1].set_title('Absolute Error along Middle Grid Line')
axes[1,1].grid()
plt.savefig('task_d/solution_lines.svg')
plt.show()

exact_field = harmonic_solution(solver.grid[1],solver.grid[0],center=boundary_condition[0,:-1],radius = boundary_condition[0,-1])
exact_field = exact_field.at[solver.inner_boundary].set(1.0)
solution_field, residual = solver.solve(outer_bc=exact_field[solver.whole_boundary], w=w, max_iter=max_iter)

plt.semilogy(residual)
plt.grid()
plt.show()

plt.imshow(jnp.abs(solution_field-exact_field),extent=[0,1,0,1], origin='lower')
plt.colorbar()
plt.show()
