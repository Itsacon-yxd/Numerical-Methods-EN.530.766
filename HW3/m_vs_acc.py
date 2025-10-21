import numpy as np
from matplotlib import pyplot as plt
from numerical_solvers import backward_euler, forward_euler, crank_nicolson,exact_solution

m_values=[3,5,7]
delta_x=np.pi/10.0
r=0.5
max_t=0.5

fig_fe,axes_fe = plt.subplots(nrows=1,ncols=len(m_values),figsize=(15,5))
fig_be,axes_be = plt.subplots(nrows=1,ncols=len(m_values),figsize=(15,5))
fig_cn,axes_cn = plt.subplots(nrows=1,ncols=len(m_values),figsize=(15,5))


for idx,m in enumerate(m_values):
    fe_solution,grid=forward_euler(m=m, delta_x=delta_x, r=r, max_t=max_t)
    be_solution,grid=backward_euler(m=m, delta_x=delta_x, r=r, max_t=max_t)
    cn_solution,grid=crank_nicolson(m=m, delta_x=delta_x, r=r, max_t=max_t)

    exact=exact_solution(grid,m=m)

    axes_fe[idx].plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    axes_fe[idx].plot(grid[0,-1],fe_solution[0,-1],label='Forward Euler',linestyle='--')
    axes_fe[idx].set_title(f'Solution at t={max_t} for m={m} for FE')
    axes_fe[idx].set_xlabel('x')
    axes_fe[idx].set_ylabel('u(x,t)')
    axes_fe[idx].legend()
    axes_fe[idx].grid()
    # fig_fe.savefig(f'm_vs_acc_d/solution_m={m}_FE.png')

    axes_be[idx].plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    axes_be[idx].plot(grid[0,-1],be_solution[0,-1],label='Backward Euler',linestyle='-.')
    axes_be[idx].set_title(f'Solution at t={max_t} for m={m} for BE')
    axes_be[idx].set_xlabel('x')
    axes_be[idx].set_ylabel('u(x,t)')
    axes_be[idx].legend()
    axes_be[idx].grid()
    # fig_be.savefig(f'm_vs_acc_d/solution_m={m}_BE.png')
    plt.show()

    axes_cn[idx].plot(grid[0,-1],exact[0,-1],label='Exact',color='black',linewidth=2)
    axes_cn[idx].plot(grid[0,-1],cn_solution[0,-1],label='Crank-Nicolson',linestyle=':')
    axes_cn[idx].set_title(f'Solution at t={max_t} for m={m} for CN')
    axes_cn[idx].set_xlabel('x')
    axes_cn[idx].set_ylabel('u(x,t)')
    axes_cn[idx].legend()
    axes_cn[idx].grid()
    # plt.savefig(f'm_vs_acc_d/solution_m={m}_CN.png')
    plt.show()

fig_fe.suptitle('Forward Euler Solutions for Different m Values')
fig_fe.savefig('m_vs_acc_d/FE_solutions.png')
fig_be.suptitle('Backward Euler Solutions for Different m Values')
fig_be.savefig('m_vs_acc_d/BE_solutions.png')
fig_cn.suptitle('Crank-Nicolson Solutions for Different m Values')
fig_cn.savefig('m_vs_acc_d/CN_solutions.png')

def amplitude_factor_fe(r, theta):
    return 1 - 4 * r * (np.sin(theta / 2))**2

def amplitude_factor_be(r, theta):
    return 1 / (1 + 4 * r * (np.sin(theta / 2))**2)

def amplitude_factor_cn(r, theta):
    return (1 - 2 * r * (np.sin(theta / 2))**2) / (1 + 2 * r * (np.sin(theta / 2))**2)

def amplitude_factor_exact(r, theta):
    return np.exp(-r * theta**2)

theta = np.linspace(0, np.pi, 100)
plt.figure(figsize=(10,6))
plt.plot(theta, amplitude_factor_fe(r, theta), label='Forward Euler', linestyle='--')
plt.plot(theta, amplitude_factor_be(r, theta), label='Backward Euler', linestyle='-.')
plt.plot(theta, amplitude_factor_cn(r, theta), label='Crank-Nicolson', linestyle=':')
plt.plot(theta, amplitude_factor_exact(r, theta), label='Exact', color='black', linewidth=2)
plt.axvline(x=3*np.pi/10, color='grey', linestyle='--', label='Theta = 3π/10 (m=3)')
plt.axvline(x=5*np.pi/10, color='brown', linestyle='--', label='Theta = 5π/10 (m=5)')
plt.axvline(x=7*np.pi/10, color='green', linestyle='--', label='Theta = 7π/10 (m=7)')
plt.title('Amplification Factors vs Theta at r=0.5')
plt.xlabel('Theta')
plt.ylabel('Amplification Factor')
plt.legend()
plt.grid()
plt.savefig('m_vs_acc_d/amplification_factors.png')
plt.show()

print(f'FE Amplification Factors for r={r} at m=3,5,7:')
for m in m_values:
    print(f'  m={m}: {amplitude_factor_fe(r, delta_x * m)}')
print(f'BE Amplification Factors for r={r} at m=3,5,7:')
for m in m_values:
    print(f'  m={m}: {amplitude_factor_be(r, delta_x * m)}')
print(f'CN Amplification Factors for r={r} at m=3,5,7:')
for m in m_values:
    print(f'  m={m}: {amplitude_factor_cn(r, delta_x * m)}')
print(f'Exact Amplification Factors for r={r} at m=3,5,7:')
for m in m_values:
    print(f'  m={m}: {amplitude_factor_exact(r, delta_x * m)}')