import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_residual(solution,delta,sum=True):
    conv_weight=torch.tensor([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ],dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    res=F.conv2d(solution,conv_weight)/delta**2

    if sum:
        return torch.abs(res).sum()

    else:
        return res

def jacobi(u_init='zero', max_iter=1000, delta=torch.pi/10):
    
    x_grid = torch.arange(0, 2*torch.pi + 0.5*delta, delta)
    y_grid = torch.arange(0, 2*torch.pi + 0.5*delta, delta)

    N = x_grid.shape[0]

    # grid of shape (x y) Y X
    grid = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='xy'))

    # Initialize solution as a 4D tensor [B, C, H, W]
    solution = torch.zeros((1, 1, N, N), dtype=torch.float32)
    if u_init=='zero':
        pass
    elif u_init=='xy':
        solution += grid[0]*grid[1]  # Initial guess u_init(x,y)=x*y
    elif u_init=='random':
        solution += 2*torch.rand((1, 1, N, N), dtype=torch.float32)-1.0
    else:
        raise ValueError("u_init must be 'zero', 'xy', or 'random'")
    
    solution[:, :, :, 0] = 0.0     # Left boundary (x=0)
    solution[:, :, :, -1] = 0.0    # Right boundary (x=2pi)
    solution[:, :, 0, :] = torch.sin(2*x_grid) + torch.sin(5*x_grid) + torch.sin(7*x_grid)  # Bottom boundary (y=0)
    solution[:, :, -1, :] = 0.0    # Top boundary (y=2pi)

    conv_weight = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) * 0.25

    residual=[]

    for _ in tqdm(range(max_iter)):

        # conved will have shape (1, 1, N-2, N-2)
        conved = F.conv2d(solution, weight=conv_weight, stride=1)

        solution[:, :, 1:-1, 1:-1] = conved

        residual.append(get_residual(solution,delta))

    return grid, solution, torch.tensor(residual)


if __name__=='__main__':

    delta = torch.pi/10 
    grid,solution,residual=jacobi(u_init='random', max_iter=1000, delta=delta)

    print(f"Grid Shape: {grid.shape},Solution Shape: {solution.shape}")

    plt.plot(residual)
    plt.yscale('log')
    plt.grid()
    plt.savefig("jacobi plot residual")
    plt.close()

    plt.imshow(solution.squeeze())
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("jacobi solution.svg")
    plt.close()

    plt.imshow(get_residual(solution,delta,sum=False).squeeze())
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig("jacobi residual.svg")
    plt.close()

    