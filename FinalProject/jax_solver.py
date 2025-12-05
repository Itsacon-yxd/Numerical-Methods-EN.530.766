import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

def get_accuracy(solution, exact_solution):

    error = jnp.abs(solution - exact_solution)
    mse = jnp.mean(error**2)
    return mse

def harmonic_solution(x, y, center, radius):
    # Analytical solution for Laplace's equation with circular boundary u = 1 + log(r/r0)
    return 1 + jnp.log(jnp.sqrt((x - center[0])**2 + (y - center[1])**2) / radius)

class Pt_GS_Solver(object):
    def __init__(self, num_pts, boundary_list):
        self.num_pts = num_pts
        self.x = jnp.linspace(0, 1, num_pts)
        self.grid = jnp.stack(jnp.meshgrid(self.x, self.x, indexing='ij'))
        self.delta = self.x[1] - self.x[0]
        self.inner_boundary = self.find_boundary(boundary_list)

        self.whole_boundary = self.inner_boundary
        self.whole_boundary = self.whole_boundary.at[0, :].set(True)
        self.whole_boundary = self.whole_boundary.at[-1, :].set(True)
        self.whole_boundary = self.whole_boundary.at[:, 0].set(True)
        self.whole_boundary = self.whole_boundary.at[:, -1].set(True)

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
            residual = self.get_residual(u, sum=True)
            return u, residual

        # Run the outer loop (max_iter)
        u_final, residuals = jax.lax.scan(sweep_fn, u, jnp.arange(max_iter))
        
        return u_final, residuals

    def solve(self, outer_bc=0.0, w=1.0, max_iter=100):
        # 1. Setup (Done outside JIT to handle dynamic shapes of active_idx)
        # JAX JIT hates it when array shapes change (like the number of active pixels)
        # so we calculate indices here in eager mode.
        
        u = jax.random.uniform(jax.random.PRNGKey(0), (self.num_pts, self.num_pts), minval=-1, maxval=1)
        
        u = u.at[self.whole_boundary].set(outer_bc)
        u = u.at[self.inner_boundary].set(1.0)
        
        # Get indices of active pixels (Where mask is False)
        # We perform this outside the JIT because the *number* of points varies
        active_r, active_c = jnp.where(~self.whole_boundary)
        active_idx = jnp.stack([active_r, active_c], axis=1)

        # 2. Run the JIT-compiled Pointwise Solver
        return self._solve_core(u, active_idx, w, max_iter)

    def get_residual(self, solution, sum=True):

        kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=solution.dtype)
        kernel = kernel[None, None, :, :]
        
        res = jax.lax.conv(solution[None, None, :, :], kernel, (1, 1), 'VALID').squeeze()
        res = jax.lax.pad(res, 0.0, [(1, 1, 0), (1, 1, 0)])
        res = res / self.delta**2
        res = res.squeeze() * (~self.whole_boundary)

        if sum:
            return jnp.mean(res**2)
        else:
            return res

class Line_GS_Solver(object):
    def __init__(self, num_pts, boundary_list):
        self.num_pts = num_pts
        self.x = jnp.linspace(0, 1, num_pts)
        self.grid = jnp.stack(jnp.meshgrid(self.x, self.x, indexing='ij'))
        self.delta = self.x[1] - self.x[0]
        self.inner_boundary = self.find_boundary(boundary_list)

        self.whole_boundary = self.inner_boundary
        self.whole_boundary = self.whole_boundary.at[0, :].set(True)
        self.whole_boundary = self.whole_boundary.at[-1, :].set(True)
        self.whole_boundary = self.whole_boundary.at[:, 0].set(True)
        self.whole_boundary = self.whole_boundary.at[:, -1].set(True)

    def find_boundary(self, boundary_list):
        mask = jnp.zeros((self.num_pts, self.num_pts), dtype=bool)
        for i in range(boundary_list.shape[0]):
            center = boundary_list[i, :-1]
            radius = boundary_list[i, -1]
            dist_sq = jnp.sum((self.grid - center[:, None, None])**2, axis=0)
            mask = jnp.logical_or(mask, dist_sq <= radius**2)
        return mask

    @partial(jax.jit, static_argnames=['self', 'max_iter'])
    def _solve_core(self, u, row_ls, w, max_iter):
        
        def update_line(u, row):
            # Get mask for this row
            mask = self.whole_boundary[row, 1:-1]
            n_total = self.num_pts - 2 
            
            # --- VECTORIZED MATRIX BUILD ---
            # Main diagonal: 1.0 for boundaries, -4.0 for unknowns
            d = jnp.where(mask, 1.0, -4.0)
            
            # Identify connections between adjacent non-boundary points
            # shape of mask is (n_total,)
            # We want to check i and i+1. 
            active_links = (~mask[:-1]) & (~mask[1:])
            
            # Construct off-diagonals immediately without loops
            # du[i] connects i to i+1. Last element is 0.
            du = jnp.append(jnp.where(active_links, 1.0, 0.0), 0.0)
            
            # dl[i] connects i to i-1. First element is 0.
            # Note: In tridiagonal_solve, dl usually aligns such that dl[i] is interaction with i-1
            dl = jnp.append(0.0, jnp.where(active_links, 1.0, 0.0))
            
            # --- VECTORIZED RHS ---
            # (Your RHS logic was largely fine, but ensure explicit indexing avoids copies)
            horiz_contrib = u[row - 1, 1:-1] + u[row + 1, 1:-1]
            
            left_contrib = jnp.where(self.whole_boundary[row, :-2], u[row, :-2], 0.0)
            right_contrib = jnp.where(self.whole_boundary[row, 2:], u[row, 2:], 0.0)
            
            rhs_val = -(horiz_contrib + left_contrib + right_contrib)
            
            # Apply identity trick to RHS
            rhs = jnp.where(mask, u[row, 1:-1], rhs_val)
            
            # Solve
            u_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:,None]).squeeze()
            
            # SOR Update
            u_updated = (1 - w) * u[row, 1:-1] + w * u_new
            
            # Final mask application (ensure boundaries strictly stay fixed)
            u_final_row = jnp.where(mask, u[row, 1:-1], u_updated)
            
            # Update grid
            u = u.at[row, 1:-1].set(u_final_row)
            
            return u, None
        
        def sweep_fn(u, _):
            u, _ = jax.lax.scan(update_line, u, row_ls)
            residual = self.get_residual(u, sum=True)
            return u, residual
        
        u_final, residuals = jax.lax.scan(sweep_fn, u, jnp.arange(max_iter))
        
        return u_final, residuals

    def solve(self, outer_bc=0.0, w=1.0, max_iter=100):
        # 1. Setup (Done outside JIT to handle dynamic shapes of active_idx)
        # JAX JIT hates it when array shapes change (like the number of active pixels)
        # so we calculate indices here in eager mode.
        
        u = jax.random.uniform(jax.random.PRNGKey(0), (self.num_pts, self.num_pts), minval=-1, maxval=1)
        
        u = u.at[self.whole_boundary].set(outer_bc)
        u = u.at[self.inner_boundary].set(1.0)

        # 2. Run the JIT-compiled Pointwise Solver
        return self._solve_core(u, jnp.arange(1,self.num_pts-1), w, max_iter)

    def get_residual(self, solution, sum=True):

        kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=solution.dtype)
        kernel = kernel[None, None, :, :]
        
        res = jax.lax.conv(solution[None, None, :, :], kernel, (1, 1), 'VALID').squeeze()
        res = jax.lax.pad(res, 0.0, [(1, 1, 0), (1, 1, 0)])
        res = res / self.delta**2
        res = res.squeeze() * (~self.whole_boundary)

        if sum:
            return jnp.mean(res**2)
        else:
            return res
        

class PotentialFlow_Solver(object):
    def __init__(self, num_pts, boundary_list):
        self.num_pts = num_pts
        self.x = jnp.linspace(0, 4, num_pts)
        self.grid = jnp.stack(jnp.meshgrid(self.x, self.x, indexing='ij'))
        self.delta = self.x[1] - self.x[0]
        self.inner_boundary = self.find_boundary(boundary_list)

        self.whole_boundary = self.inner_boundary
        self.whole_boundary = self.whole_boundary.at[0, :].set(True)
        self.whole_boundary = self.whole_boundary.at[-1, :].set(True)
        self.whole_boundary = self.whole_boundary.at[:, 0].set(True)
        self.whole_boundary = self.whole_boundary.at[:, -1].set(True)

        active_r, active_c = jnp.where(~self.whole_boundary)
        self.active_idx = jnp.stack([active_r, active_c], axis=1)

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
    def _solve_core(self, u,  w, max_iter):
        
        # The Step Function: What happens for ONE pixel update
        def update_pixel(u, idx_pair, boundary=self.inner_boundary):
            r, c = idx_pair
            
            val_up    = jax.lax.cond(boundary[r-1,c], lambda _:u[r,c], lambda _:u[r-1,c], None)
            val_down  = jax.lax.cond(boundary[r+1,c], lambda _:u[r,c], lambda _:u[r+1,c], None)
            val_left  = jax.lax.cond(boundary[r,c-1], lambda _:u[r,c], lambda _:u[r,c-1], None)
            val_right = jax.lax.cond(boundary[r,c+1], lambda _:u[r,c], lambda _:u[r,c+1], None)
            
            gs_est = 0.25 * (val_up + val_down + val_left + val_right)
            new_val = (1 - w) * u[r, c] + w * gs_est
            
            # In-place update using .at[].set()
            u = u.at[r, c].set(new_val)
            return u, None # scan requires a 'carry' (u) and 'output' (None)

        # The Iteration Body: One full sweep over the grid
        def sweep_fn(u_prev, _):
            # scan loops over 'active_idx', carrying 'u' along
            u_new, _ = jax.lax.scan(update_pixel, u_prev, self.active_idx)

            diff = jnp.abs(u_new - u_prev)
            return u_new, jnp.nansum(diff)

        # Run the outer loop (max_iter)
        u_final, residual = jax.lax.scan(sweep_fn, u, jnp.arange(max_iter))
        
        return u_final, residual

    def solve(self, w=1.0, max_iter=100):
        
        u = jax.random.uniform(jax.random.PRNGKey(0), (self.num_pts, self.num_pts), minval=-1, maxval=1)
        
        u = u.at[self.whole_boundary].set(self.grid[1][self.whole_boundary])
        u = u.at[self.inner_boundary].set(jnp.nan)

        return self._solve_core(u, w, max_iter)
    
    def get_velocity_field(self, solution):


        velocity = jnp.zeros((2,self.num_pts,self.num_pts))

        def update_pixel(velocity, idx_pair, u=solution, boundary=self.inner_boundary):
            r, c = idx_pair
            
            val_up    = jax.lax.cond(boundary[r-1,c], lambda _:u[r,c], lambda _:u[r-1,c], None)
            val_down  = jax.lax.cond(boundary[r+1,c], lambda _:u[r,c], lambda _:u[r+1,c], None)
            val_left  = jax.lax.cond(boundary[r,c-1], lambda _:u[r,c], lambda _:u[r,c-1], None)
            val_right = jax.lax.cond(boundary[r,c+1], lambda _:u[r,c], lambda _:u[r,c+1], None)

            u_x = (val_right - val_left) / (2*self.delta)
            u_y = (val_up - val_down) / (2*self.delta)

            velocity = velocity.at[0,r,c].set(u_x)
            velocity = velocity.at[1,r,c].set(u_y)
            
            return velocity, None # scan requires a 'carry' (u) and 'output' (None)
            
        velocity, _ = jax.lax.scan(update_pixel, velocity, self.active_idx)

        # --- Top Edge (Row 0) ---
        velocity = velocity.at[1, 0, :].set((solution[0, :] - solution[1, :]) / self.delta) 
        velocity = velocity.at[0, 0, 1:-1].set((solution[0, 2:] - solution[0, :-2]) / (2*self.delta))

        # --- Bottom Edge (Row -1) ---
        velocity = velocity.at[1, -1, :].set((solution[-2, :] - solution[-1, :]) / self.delta)
        velocity = velocity.at[0, -1, 1:-1].set((solution[-1, 2:] - solution[-1, :-2]) / (2*self.delta))

        # --- Left Edge (Col 0) ---
        velocity = velocity.at[0, :, 0].set((solution[:, 1] - solution[:, 0]) / self.delta)
        velocity = velocity.at[1, 1:-1, 0].set((solution[2:, 0] - solution[:-2, 0]) / (2*self.delta))

        # --- Right Edge (Col -1) ---
        velocity = velocity.at[0, :, -1].set((solution[:, -1] - solution[:, -2]) / self.delta)
        velocity = velocity.at[1, 1:-1, -1].set((solution[2:, -1] - solution[:-2, -1]) / (2*self.delta))

        return velocity

# --- Comparison Script ---
if __name__ == "__main__":
    N = 256 # Grid Size
        
    boundary_list = jnp.array([
        [0.5, 0.5, 0.25],
        # [0.25, 0.25, 0.125],
        # [0.25, 0.75, 0.125],
        # [0.75, 0.25, 0.125],
        # [0.75, 0.75, 0.125]
    ])
    
    print(f"Initializing Pointwise JAX Solver (N={N})...")
    jax_solver = Pt_GS_Solver(num_pts=N, boundary_list=boundary_list)

    print("Compiling (Warmup)...")
    # This might take a moment because unrolling/scanning over pixels is heavy
    start_compile = time.time()
    _ = jax_solver.solve(w=1.5, max_iter=2) 
    print(f"Compilation finished in {time.time() - start_compile:.2f}s")
    
    print("Running Solve...")
    start = time.time()
    solution, residuals = jax_solver.solve(w=1.5, max_iter=20000)
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