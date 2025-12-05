from jax_solver import Line_GS_Solver
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
N = 224 # Grid Size
    
boundary_list = jnp.array([
    [0.5, 0.5, 0.25],
    # [0.25, 0.25, 0.125],
    # [0.25, 0.75, 0.125],
    # [0.75, 0.25, 0.125],
    # [0.75, 0.75, 0.125]
])

print(f"Initializing Pointwise JAX Solver (N={N})...")
jax_solver = Line_GS_Solver(num_pts=N, boundary_list=boundary_list)

print("Compiling (Warmup)...")
# This might take a moment because unrolling/scanning over pixels is heavy
start_compile = time.time()
_ = jax_solver.solve(w=1.5, max_iter=2) 
print(f"Compilation finished in {time.time() - start_compile:.2f}s")

print("Running Solve...")
start = time.time()
solution,residuals = jax_solver.solve(w=1.0, max_iter=20000)
solution.block_until_ready()  # Ensure computation is done
end = time.time()

print(f"JAX Pointwise Time: {end - start:.4f} seconds")

# Plot
plt.imshow(solution, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()

plt.imshow(jax_solver.get_residual(solution, sum=False), origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()

plt.plot(residuals)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Residual')
plt.grid()
plt.show()