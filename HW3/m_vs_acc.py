import numpy as np
from matplotlib import pyplot as plt
from numerical_solvers import backward_euler, forward_euler, crank_nicolson,exact_solution

m_values=[3,5,7]
delta_x=np.pi/10.0
r=0.5
max_t=0.5

for m in m_values:
    fe_solution,grid=forward_euler(m=m, delta_x=delta_x, r=r, max_t=max_t)
    be_solution,grid=backward_euler(m=m, delta_x=delta_x, r=r, max_t=max_t)
    cn_solution,grid=crank_nicolson(m=m, delta_x=delta_x, r=r, max_t=max_t)

    exact=exact_solution(grid,m=m)

    plt.figure(figsize=(10,6))
    plt.plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    plt.plot(grid[0,-1],fe_solution[0,-1],label='Forward Euler',linestyle='--')
    plt.title(f'Solution at t={max_t} for m={m} for FE')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid()
    plt.savefig(f'm_vs_acc_d/solution_m={m}_FE.png')
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    plt.plot(grid[0,-1],be_solution[0,-1],label='Backward Euler',linestyle='-.')
    plt.title(f'Solution at t={max_t} for m={m} for BE')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid()
    plt.savefig(f'm_vs_acc_d/solution_m={m}_BE.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    plt.plot(grid[0,-1],cn_solution[0,-1],label='Crank-Nicolson',linestyle=':')
    plt.title(f'Solution at t={max_t} for m={m} for CN')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid()
    plt.savefig(f'm_vs_acc_d/solution_m={m}_CN.png')
    plt.show()
    
def amplitude_factor_fe(r, theta):
    return 1 - 4 * r * (np.sin(theta / 2))**2

def amplitude_factor_be(r, theta):
    return 1 / (1 + 4 * r * (np.sin(theta / 2))**2)

def amplitude_factor_cn(r, theta):
    return (1 - 2 * r * (np.sin(theta / 2))**2) / (1 + 2 * r * (np.sin(theta / 2))**2)

theta = np.linspace(0, np.pi, 100)
plt.figure(figsize=(10,6))
plt.plot(theta, amplitude_factor_fe(r, theta), label='Forward Euler', linestyle='--')
plt.plot(theta, amplitude_factor_be(r, theta), label='Backward Euler', linestyle='-.')
plt.plot(theta, amplitude_factor_cn(r, theta), label='Crank-Nicolson', linestyle=':')
plt.axvline(x=3*np.pi/10, color='grey', linestyle='--', label='Theta = 3π/10 (m=3)')
plt.axvline(x=5*np.pi/10, color='brown', linestyle='--', label='Theta = 5π/10 (m=5)')
plt.axvline(x=7*np.pi/10, color='green', linestyle='--', label='Theta = 7π/10 (m=7)')
plt.title('Amplitude Factors vs Theta')
plt.xlabel('Theta')
plt.ylabel('Amplitude Factor')
plt.legend()
plt.grid()
plt.savefig('m_vs_acc_d/amplitude_factors.png')
plt.show()