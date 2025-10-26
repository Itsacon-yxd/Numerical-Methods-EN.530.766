import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import backward_euler, crank_nicolson, exact_solution

delta_x=0.1

r_values= [0.4,0.6,0.8,1.0] #np.linspace(0.5,0.6,20)
max_t=1.0


def l_1_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

be_acc,cn_acc = [],[]
# %%
fig,axs=plt.subplots(2,2,figsize=(12,10))
axs=axs.flatten()
for idxr, r in enumerate(r_values):
    be_solution,grid=backward_euler(max_t=max_t, delta_x=delta_x, r=r,m=2)
    cn_solution,grid=crank_nicolson(max_t=max_t, delta_x=delta_x, r=r,m=2)
    exact = exact_solution(grid,m=2)
    be_acc.append(l_1_loss(exact[:,-1], be_solution[:,-1]))
    cn_acc.append(l_1_loss(exact[:,-1], cn_solution[:,-1]))

    axs[idxr].plot(grid[0, -1], exact[0, -1], label='Exact Solution', color='black', linewidth=2)
    axs[idxr].plot(grid[0, -1], be_solution[0, -1], label='Backward Euler', linestyle='--')
    axs[idxr].plot(grid[0, -1], cn_solution[0, -1], label='Crank-Nicolson', linestyle=':')
    axs[idxr].set_xlabel('x')
    axs[idxr].set_ylabel('u(x, t)')
    axs[idxr].set_title(f'Solution at t={max_t} for r={r}')
    axs[idxr].legend(loc='lower left')
    axs[idxr].grid()
plt.savefig(f'r_vs_accuracy_c_ii_c/r_implicit.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(r_values, be_acc, label='Backward Euler', color='blue')
plt.plot(r_values, cn_acc, label='Crank-Nicolson', color='orange')
plt.xlabel('r values')
plt.ylabel('L1 Accuracy')
plt.yscale('log')
plt.title('L1 Error vs r values for Implicit Methods')
plt.legend()
plt.grid()
plt.savefig('r_vs_accuracy_c_ii_c/r_vs_acc_implicit.png')
plt.show()