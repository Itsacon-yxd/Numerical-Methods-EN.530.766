import torch
import matplotlib.pyplot as plt
from elliptic_solvers import sor_jacobi, sor_gauss_seidel
from joblib import Parallel, delayed

# --- Configuration ---
torch.manual_seed(42)
delta = torch.pi / 10
criterion = 1e-4
max_iter = 2000

# --- 1. Define the Worker Function ---
def solve_instance(solver_func, w, max_iter, delta, criterion):
    """
    Helper function to run a single solver instance.
    """
    # IMPORTANT: Limit threads per process to avoid CPU thrashing
    torch.set_num_threads(1) 
    
    # Run the solver
    grid, solution, residual = solver_func(
        u_init='xy',
        max_iter=max_iter,
        delta=delta,
        w=w
    )
    torch.set_num_threads(torch.get_num_threads())  # Reset to default after computation
    # Check convergence
    converg = torch.where(residual < criterion)[0]
    check_instability = torch.
    if len(converg) > 0 and 
    # return converg[0].item() if len(converg) > 0 else max_iter

# --- 2. Main Execution Block ---
# (Necessary for multiprocessing on Windows/macOS)
if __name__ == "__main__":
    
    # Define ranges
    jac_w_list = torch.linspace(0.5, 1.1, steps=100)
    gs_w_list = torch.linspace(0.9, 2.1, steps=20)

    print("Running Jacobi Parallel Simulation...")
    # n_jobs=-1 uses all available CPU cores
    jac_converg_list = Parallel(n_jobs=-1)(
        delayed(solve_instance)(sor_jacobi, w.item(), max_iter, delta, criterion) 
        for w in jac_w_list
    )

    # Plot Jacobi
    plt.figure(1)
    plt.plot(jac_w_list, jac_converg_list)
    plt.xlabel('Relaxation Factor w')
    plt.ylabel('Iterations to Convergence')
    plt.title('Convergence Behavior of SOR Jacobi vs Relaxation Factor')
    plt.grid()
    plt.savefig('p2/sor_jacobi_convergence.svg')
    # plt.show() # Commented out to allow second plot to generate without blocking

    print("Running Gauss-Seidel Parallel Simulation...")
    gs_converg_list = Parallel(n_jobs=-1)(
        delayed(solve_instance)(sor_gauss_seidel, w.item(), max_iter, delta, criterion) 
        for w in gs_w_list
    )

    # Plot Gauss-Seidel
    plt.figure(2)
    plt.plot(gs_w_list, gs_converg_list)
    plt.xlabel('Relaxation Factor w')
    plt.ylabel('Iterations to Convergence')
    plt.title('Convergence Behavior of SOR Gauss-Seidel vs Relaxation Factor')
    plt.grid()
    plt.savefig('p2/sor_gauss_seidel_convergence.svg')
    
    plt.show() # Show both

    print("SOR Jacobi Best w:", jac_w_list[torch.argmin(torch.tensor(jac_converg_list))].item())
    print("SOR Gauss-Seidel Best w:", gs_w_list[torch.argmin(torch.tensor(gs_converg_list))].item())