# %%
import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import forward_euler, backward_euler, crank_nicolson, exact_solution

# %% [markdown]
# PDE:
# $$
# u_t=u_{xx},\quad0\le x\le2\pi
# $$
# Boundary Conditions:
# $$
# u(0,t)=u(2\pi,t)=0
# $$
# Initial Condition:
# $$
# u(x,0)=\sin(mx)
# $$
# Exact Solution:
# $$
# u=\sin(mx)\exp(-m^2t)
# $$

# %%
fe_solution,grid=forward_euler(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
be_solution,grid=backward_euler(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
cn_solution,grid=crank_nicolson(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
print(grid.shape) #(x t) time space

# %%


# %%
solution = exact_solution(grid, 2)

# %%
t_01_ind=np.argmin(np.abs(grid[1,:,0]-0.1))
t_05_ind=np.argmin(np.abs(grid[1,:,0]-0.5))
t_1_ind=np.argmin(np.abs(grid[1,:,0]-1))
print(f't_0.1 index: {t_01_ind}, t_0.5 index: {t_05_ind}, t_1 index: {t_1_ind}, t_0.1 time: {grid[1,t_01_ind,0]}, t_0.5 time: {grid[1,t_05_ind,0]}, t_1 time: {grid[1,t_1_ind,0]}')

# %%
fig=plt.figure(figsize=(10,6))
plt.plot(grid[0,t_01_ind], solution[0,t_01_ind], label='Exact Solution',c='b')
plt.plot(grid[0,t_01_ind], fe_solution[0,t_01_ind], label='FE Solution',c='r')
plt.plot(grid[0,t_01_ind], be_solution[0,t_01_ind], label='BE Solution',c='g')
plt.plot(grid[0,t_01_ind], cn_solution[0,t_01_ind], label='CN Solution',c='y')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Heat Equation Solutions at t=0.1')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('time_snapshots_c_i/heat_conduction_solutions_t=0.1s.png')
plt.show()

# %%
fig=plt.figure(figsize=(10,6))
plt.plot(grid[0,t_05_ind], solution[0,t_05_ind], label='Exact Solution',c='b')
plt.plot(grid[0,t_05_ind], fe_solution[0,t_05_ind], label='FE Solution',c='r')
plt.plot(grid[0,t_05_ind], be_solution[0,t_05_ind], label='BE Solution',c='g')
plt.plot(grid[0,t_05_ind], cn_solution[0,t_05_ind], label='CN Solution',c='y')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Heat Equation Solutions at t=0.5')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('time_snapshots_c_i/heat_conduction_solutions_t=0.5s.png')
plt.show()

# %%
fig=plt.figure(figsize=(10,6))
plt.plot(grid[0,t_1_ind], solution[0,t_1_ind], label='Exact Solution',c='b')
plt.plot(grid[0,t_1_ind], fe_solution[0,t_1_ind], label='FE Solution',c='r')
plt.plot(grid[0,t_1_ind], be_solution[0,t_1_ind], label='BE Solution',c='g')
plt.plot(grid[0,t_1_ind], cn_solution[0,t_1_ind], label='CN Solution',c='y')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Heat Equation Solutions at t=1')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('time_snapshots_c_i/heat_conduction_solutions_t=1s.png')
plt.show()