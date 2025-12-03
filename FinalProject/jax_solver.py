import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

class LaplaceSolverJAXPointwise(object):
    def __init__(self, num_pts):
        self.num_pts = num_pts
        self.x = jnp.linspace(0, 1, num_pts)
        self.grid = jnp.stack(jnp.meshgrid(self.x, self.x, indexing='ij'))
        self.delta = self.x[1] - self.x[0]

    def find_boundary(self, boundary_list):
        mask = jnp.zeros((self.num_pts, self.num_pts), dtype=bool)
        for i in range(boundary_list.shape[0]):
            center = boundary_list[i, :-1]
            radius = boundary_list[i, -1]
            dist_sq = jnp.sum((self.grid - center[:, None, None])**2, axis=0)
            mask = jnp.logical_or(mask, dist_sq <= radius**2)
        return mask

    # We separate the "Solver Core" to JIT compile just the loop part.
    # We must pass 'active_idx' in because JAX needs to know loop lengths at compile time.
    @partial(jax.jit, static_argnames=['self', 'max_iter'])
    def _solve_core(self, u, active_idx, w, max_iter):
        
        # The Step Function: What happens for ONE pixel update
        def update_pixel(u, idx_pair):
            r, c = idx_pair
            
            # Read neighbors
            # Note: We must explicitly handle boundaries or assume active_idx 
            # excludes edges (which your logic does).
            val_up    = u[r-1, c]
            val_down  = u[r+1, c]
            val_left  = u[r, c-1]
            val_right = u[r, c+1]
            
            gs_est = 0.25 * (val_up + val_down + val_left + val_right)
            new_val = (1 - w) * u[r, c] + w * gs_est
            
            # In-place update using .at[].set()
            u = u.at[r, c].set(new_val)
            return u, None # scan requires a 'carry' (u) and 'output' (None)

        # The Iteration Body: One full sweep over the grid
        def sweep_fn(u, _):
            # scan loops over 'active_idx', carrying 'u' along
            u, _ = jax.lax.scan(update_pixel, u, active_idx)
            return u, None

        # Run the outer loop (max_iter)
        u_final, _ = jax.lax.scan(sweep_fn, u, jnp.arange(max_iter))
        
        return u_final

    def solve(self, boundary_list, w=1.0, max_iter=100):
        # 1. Setup (Done outside JIT to handle dynamic shapes of active_idx)
        # JAX JIT hates it when array shapes change (like the number of active pixels)
        # so we calculate indices here in eager mode.
        
        u = 2 * jax.random.uniform(jax.random.PRNGKey(0), (self.num_pts, self.num_pts)) - 1
        inner_boundary = self.find_boundary(boundary_list)
        
        whole_boundary = inner_boundary
        whole_boundary = whole_boundary.at[0, :].set(True)
        whole_boundary = whole_boundary.at[-1, :].set(True)
        whole_boundary = whole_boundary.at[:, 0].set(True)
        whole_boundary = whole_boundary.at[:, -1].set(True)
        
        u = u.at[whole_boundary].set(0.0)
        u = u.at[inner_boundary].set(1.0)
        
        # Get indices of active pixels (Where mask is False)
        # We perform this outside the JIT because the *number* of points varies
        active_r, active_c = jnp.where(~whole_boundary)
        active_idx = jnp.stack([active_r, active_c], axis=1)

        # 2. Run the JIT-compiled Pointwise Solver
        return self._solve_core(u, active_idx, w, max_iter)

    def get_residual(self, solution, boundary, sum=True):

        kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32)
        kernel = kernel[None, None, :, :]
        
        res = jax.lax.conv(solution[None, None, :, :], kernel, (1, 1), 'VALID').squeeze()
        res = jax.lax.pad(res, 0.0, [(1, 1, 0), (1, 1, 0)])
        res = res / self.delta**2
        res = res.squeeze() * (~boundary)

        if sum:
            return jnp.mean(res**2)
        else:
            return res

# --- Comparison Script ---
if __name__ == "__main__":
    N = 224 # Kept small because pointwise JAX is very slow
    
    print(f"Initializing Pointwise JAX Solver (N={N})...")
    jax_solver = LaplaceSolverJAXPointwise(num_pts=N)
    
    boundary_list = jnp.array([
        [0.25, 0.25, 0.125],
        [0.25, 0.75, 0.125],
        [0.75, 0.25, 0.125],
        [0.75, 0.75, 0.125]
    ])
    mask = jax_solver.find_boundary(boundary_list)
    print("Compiling (Warmup)...")
    # This might take a moment because unrolling/scanning over pixels is heavy
    start_compile = time.time()
    _ = jax_solver.solve(boundary_list, w=1.5, max_iter=2) 
    print(f"Compilation finished in {time.time() - start_compile:.2f}s")
    
    print("Running Solve...")
    start = time.time()
    solution = jax_solver.solve(boundary_list, w=1.5, max_iter=20000).block_until_ready()
    end = time.time()
    
    print(f"JAX Pointwise Time: {end - start:.4f} seconds")

    # Plot
    plt.imshow(solution, origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    plt.imshow(jax_solver.get_residual(solution, mask, sum=False), origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()