import jax
from jax_solver import PotentialFlow_Solver
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

boundary_list = 4*jnp.array([
        # [0.5, 0.5, 0.25],
        [0.25, 0.25, 0.125],
        [0.25, 0.75, 0.125],
        [0.75, 0.25, 0.125],
        [0.75, 0.75, 0.125]
])

solver = PotentialFlow_Solver(num_pts = 224, boundary_list=boundary_list)
_ = solver.solve(max_iter=2)

solution, converge = solver.solve(w = 1.5, max_iter=70000)

plt.imshow(solution,extent=[0,4,0,4],origin='lower')
plt.colorbar()
plt.show()
plt.semilogy(converge)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.show()

velocity = np.array(solver.get_velocity_field(solution))
plt.streamplot(
    np.array(solver.grid[1]), 
    np.array(solver.grid[0]), 
    velocity[0], 
    velocity[1], 
    density=2
    )
plt.title('Velocity Field Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,4)
plt.ylim(0,4)
plt.show()

pressure = 0.5 - 0.5 * (velocity[0]**2 + velocity[1]**2)
pressure[solver.inner_boundary] = np.nan
plt.imshow(pressure,extent=[0,4,0,4],origin='lower')
plt.colorbar()
plt.show()