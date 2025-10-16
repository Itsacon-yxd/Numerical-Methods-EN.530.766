# %%
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

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
def forward_sweep(A,b):
    A_prime=deepcopy(A)
    b_prime=deepcopy(b)
    A_prime[0]/=A[0,0]
    b_prime[0]/=A[0,0]
    for i in range(1,A.shape[0]):
        b_prime[i]=(b_prime[i]-A_prime[i,i-1]/A_prime[i-1,i-1]*b_prime[i-1])/(A_prime[i,i]-A_prime[i,i-1]/A_prime[i-1,i-1]*A_prime[i-1,i])
        
        A_prime[i]=(A_prime[i]-A_prime[i,i-1]/A_prime[i-1,i-1]*A_prime[i-1])/(A_prime[i,i]-A_prime[i,i-1]/A_prime[i-1,i-1]*A_prime[i-1,i])
    
    return A_prime,b_prime
def solve_tridiagonal(A,b):
    A_prime,b_prime=forward_sweep(A,b)
    x=np.zeros_like(b)
    x[-1]=b_prime[-1]
    for i in reversed(range(b.shape[0]-1)):
        x[i]=b_prime[i]-A_prime[i,i+1]*x[i+1]
    return x

# %%
def exact_solution(grid,m):
    u=np.zeros((1,grid.shape[1],grid.shape[2]))
    for i in range(u.shape[1]):
        u[:,i,:]=np.exp(-m**2 * grid[1,i]) * np.sin(m * grid[0,i])
    return u
# %%
def forward_euler(max_t,m,r=None,delta_x=None,delta_t=None):
    if delta_t is None:
        delta_t=r*delta_x**2
    if delta_x is None:
        delta_x=np.sqrt(delta_t/r)
    if r is None:
        r=delta_t/delta_x**2
    grid=np.array(np.meshgrid(np.arange(0,2*np.pi+delta_x/2,delta_x),np.arange(0,max_t+delta_t/2,delta_t)))
    u=np.zeros((1,grid.shape[1],grid.shape[2])) #time space
    u[:,0,:]=np.sin(m * grid[0,0]) #initial condition
    u[:,:,0]=0 #boundary condition
    u[:,:,-1]=0 #boundary condition
    for n in range(u.shape[1]-1): #time
        for j in range(1,u.shape[2]-1): #space
            u[:,n+1,j]=r*(u[:,n,j+1]-2*u[:,n,j]+u[:,n,j-1])+u[:,n,j]
    return u,grid

# %%
def backward_euler(max_t,m,r=None,delta_x=None,delta_t=None):
    if delta_t is None:
        delta_t=r*delta_x**2
    if delta_x is None:
        delta_x=np.sqrt(delta_t/r)
    if r is None:
        r=delta_t/delta_x**2
    grid=np.array(np.meshgrid(np.arange(0,2*np.pi+delta_x/2,delta_x),np.arange(0,max_t+delta_t/2,delta_t)))
    u=np.zeros((1,grid.shape[1],grid.shape[2])) #time space
    
    u[:,0,:]=np.sin(m * grid[0,0]) #initial condition
    u[:,:,0]=0 #boundary condition
    u[:,:,-1]=0 #boundary condition
    
    A=np.zeros((u.shape[2]-2,u.shape[2]-2))
    np.fill_diagonal(A,1+2*r)
    np.fill_diagonal(A[1:],-r)
    np.fill_diagonal(A[:,1:],-r)

    for n in range(u.shape[1]-1): #time
        b=u[:,n,1:-1].T
        u[:,n+1,1:-1]=solve_tridiagonal(A,b).T
    return u,grid

# %%
def crank_nicolson(max_t,m,r=None,delta_x=None,delta_t=None):
    if delta_t is None:
        delta_t=r*delta_x**2
    if delta_x is None:
        delta_x=np.sqrt(delta_t/r)
    if r is None:
        r=delta_t/delta_x**2
    
    grid=np.array(np.meshgrid(np.arange(0,2*np.pi+delta_x/2,delta_x),np.arange(0,max_t+delta_t/2,delta_t)))
    
    u=np.zeros((1,grid.shape[1],grid.shape[2])) #time space
    u[:,0,:]=np.sin(m * grid[0,0]) #initial condition
    u[:,:,0]=0 #boundary condition
    u[:,:,-1]=0 #boundary condition
    
    A=np.zeros((u.shape[2]-2,u.shape[2]-2))
    np.fill_diagonal(A,2+2*r)
    np.fill_diagonal(A[1:],-r)
    np.fill_diagonal(A[:,1:],-r)
    
    B=np.zeros((u.shape[2]-2,u.shape[2]-2))
    np.fill_diagonal(B,2-2*r)
    np.fill_diagonal(B[1:],r)
    np.fill_diagonal(B[:,1:],r)

    for n in range(u.shape[1]-1): #time
        b=B @ u[:,n,1:-1].T
        u[:,n+1,1:-1]=solve_tridiagonal(A,b).T
    return u,grid


if __name__ == "__main__":
    fe_solution,grid=forward_euler(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
    be_solution,grid=backward_euler(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
    cn_solution,grid=crank_nicolson(delta_x=2*np.pi/20,max_t=2,r=1/3,m=2)
    print(grid.shape) #(x t) time space

    # %%
    solution = exact_solution(grid, 2)
    plt.plot(grid[0,0], solution[0,-1], label='Exact', color='black', linewidth=2)
    plt.plot(grid[0,0], fe_solution[0,-1], label='Forward Euler', color='orange', linestyle='--')
    plt.plot(grid[0,0], be_solution[0,-1], label='Backward Euler', color='blue', linestyle='-.')
    plt.plot(grid[0,0], cn_solution[0,-1], label='Crank-Nicolson', color='green', linestyle=':')
    plt.xlabel('x')
    plt.ylabel('u(x, t=2)')
    plt.title('Numerical and Exact Solutions at t=2')
    plt.legend()
    plt.grid()
    plt.savefig('numerical_vs_exact_at_t=2.png')
    plt.show()
