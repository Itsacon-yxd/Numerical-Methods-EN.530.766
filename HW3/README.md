# HW3 Solutions — Numerical Methods for Heat Equation

This directory contains solutions to Homework 3 (HW3_F25) for EN.530.766 Numerical Methods. The assignment focuses on implementing and analyzing numerical methods for solving the 1D heat (diffusion) equation:

$$u_t = u_{xx}, \quad 0 \le x \le 2\pi$$

with homogeneous Dirichlet boundary conditions $u(0,t) = u(2\pi,t) = 0$ and sinusoidal initial condition $u(x,0) = \sin(mx)$.

The exact solution is: $u(x,t) = \sin(mx)\exp(-m^2t)$

## Files

### Core Implementation
- **`numerical_solvers.py`** — Core module containing implementations of:
  - `forward_euler()` — Explicit Forward Euler method
  - `backward_euler()` — Implicit Backward Euler method (uses tridiagonal solver)
  - `crank_nicolson()` — Implicit Crank-Nicolson method (2nd order in time)
  - `solve_tridiagonal()` — Thomas algorithm for solving tridiagonal systems
  - `exact_solution()` — Computes the analytical solution for validation

### Analysis Scripts

#### Part C(i): Time Snapshots
- **`visualize_time_snapshots.py`** — Generates comparison plots of all three numerical methods against the exact solution at t = 0.1, 0.5, and 1.0 seconds
  - Output: `time_snapshots_c_i/heat_conduction_solutions_t={time}.png`

#### Part C(ii)(a): Grid Refinement Study
- **`grid_refinement_study.py`** — Spatial convergence analysis with fixed small temporal step
  - Tests spatial grid refinement with δt = 1e-7 (very small to isolate spatial error)
  - Varies δx from 2e-3 to ~1.024 (powers of 2)
  - Plots L1 error vs δx with O(δx²) reference line for each method
  - Output: `grid_refinement_plots_c_ii_a/heat_conduction_L1_error_{FE,BE,CN}.png`

#### Part C(ii)(b): Stability Analysis
- **`stability_analysis.py`** — Forward Euler stability study across the CFL condition boundary
  - Tests r values: [0.1, 0.25, 0.49, 0.51, 0.75, 1.0]
  - Shows stable behavior (r ≤ 0.5) vs unstable behavior (r > 0.5)
  - Output: 2×3 subplot grid in `stability_analysis_plots_c_ii_b/`

#### Part C(ii)(c): Implicit Methods with Large Time Steps
- **`r_vs_acc_implicit.py`** — Tests Backward Euler and Crank-Nicolson with r > 0.5
  - Compares BE and CN accuracy for r = [0.4, 0.6, 0.8, 1.0]
  - Shows that implicit methods remain stable even with large time steps
  - Output: `r_vs_accuracy_c_ii_c/r_implicit.png` and `r_vs_acc_implicit.png`

#### Part D: Mode Number Effects
- **`m_vs_acc.py`** — Studies how different wavenumbers (m = 3, 5, 7) affect accuracy
  - Compares numerical solutions for different initial condition frequencies
  - Plots amplification factors vs θ for all methods
  - Shows that higher modes (larger m) are more challenging to resolve accurately
  - Output: `m_vs_acc_d/` containing individual method plots and amplification factor comparison

## Dependencies

```bash
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Running the Code

Each script can be run independently. From the HW3 directory:

```bash
# Part C(i): Generate time snapshot comparisons
python visualize_time_snapshots.py

# Part C(ii)(a): Spatial convergence study
python grid_refinement_study.py

# Part C(ii)(b): Stability analysis for Forward Euler
python stability_analysis.py

# Part C(ii)(c): Implicit methods with large r
python r_vs_acc_implicit.py

# Part D: Mode number effects
python m_vs_acc.py
```

Or run all experiments:
```bash
python visualize_time_snapshots.py && \
python grid_refinement_study.py && \
python stability_analysis.py && \
python r_vs_acc_implicit.py && \
python m_vs_acc.py
```

## Key Results

### Stability
- **Forward Euler**: Conditionally stable, requires r = Δt/Δx² ≤ 0.5 (CFL condition)
- **Backward Euler**: Unconditionally stable (can use any r > 0)
- **Crank-Nicolson**: Unconditionally stable (can use any r > 0)

### Accuracy
- **Forward Euler**: O(Δt, Δx²) — first-order in time, second-order in space
- **Backward Euler**: O(Δt, Δx²) — first-order in time, second-order in space
- **Crank-Nicolson**: O(Δt², Δx²) — second-order in both time and space

### Observations
- Higher mode numbers (larger m) are more difficult to resolve accurately
- Crank-Nicolson provides the best accuracy for a given computational cost
- Implicit methods (BE, CN) allow larger time steps without stability issues
- All methods show O(Δx²) spatial convergence when temporal error is negligible

## Output Directories

Generated plots are organized into subdirectories corresponding to homework problems:
- `time_snapshots_c_i/` — Part C(i) visualizations
- `grid_refinement_plots_c_ii_a/` — Part C(ii)(a) convergence plots
- `stability_analysis_plots_c_ii_b/` — Part C(ii)(b) stability analysis
- `r_vs_accuracy_c_ii_c/` — Part C(ii)(c) implicit method comparisons
- `m_vs_acc_d/` — Part D mode number studies

## Notes

- The grid structure returned by solvers is `grid[0]` = x-coordinates, `grid[1]` = t-coordinates
- Solutions are stored as `u[0, time_index, space_index]`
- L1 error is computed as the mean absolute difference from the exact solution
- All plotting functions use proper Axes methods (`.set_title()`, `.set_xlabel()`, `.set_ylabel()`)
