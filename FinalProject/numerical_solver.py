# %%
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# %%
class LaplaceSolver(object):
    def __init__(self, num_pts):
        self.x = torch.linspace(0,1,steps=num_pts)
        self.grid = torch.stack(torch.meshgrid(self.x,self.x,indexing='ij'))  # 2 x num_pts x num_pts
        self.num_pts = num_pts
        self.delta = self.x[1]-self.x[0]

    def find_boundary(self, boundary_list):
        mask = torch.zeros((self.num_pts, self.num_pts), dtype=torch.bool)

        for boundary in boundary_list:
            
            center = boundary[:-1]
            radius = boundary[-1]
            mask = torch.logical_or(mask, torch.sum((self.grid - center[:, None, None])**2, dim=0) <= radius**2)

        return mask
    
    def solve(self, boundary_list, w=1.0, max_iter=100, verbose=True):
        u = 2 * torch.rand(self.num_pts, self.num_pts) - 1
        
        inner_boundary = self.find_boundary(boundary_list)
        
        # Create a "Total Fixed Mask" (Obstacles + Outer Frame)
        # We need to lock the outer edges (Dirichlet BCs) and the internal obstacles
        whole_boundary = inner_boundary.clone()
        whole_boundary[0, :] = True  # Bottom
        whole_boundary[-1, :] = True # Top
        whole_boundary[:, 0] = True  # Left
        whole_boundary[:, -1] = True # Right
        
        u[whole_boundary] = 0
        
        u[inner_boundary] = 1 

        active_idx = torch.stack(torch.where(~whole_boundary)).T
        
        for _ in tqdm(range(max_iter), disable=not verbose):
                    
            for row, col in active_idx:
                gs_est = 0.25 * (u[row-1, col] + u[row, col-1] + 
                                        u[row+1, col] + u[row, col+1])
                        
                u[row, col] = (1 - w) * u[row, col] + w * gs_est
                        
        return u
    
    def get_residual(self,solution,boundary,sum=True):

        conv_weight=torch.tensor([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ],dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        res=F.conv2d(solution[None,None,:,:],conv_weight)/self.delta**2
        res=F.pad(res,pad=[1,1,1,1],mode='constant',value=0)

        res *= ~boundary

        if sum:
            return F.mse_loss(res,torch.zeros_like(res))

        else:
            return res.squeeze()



if __name__ == "__main__":
    solver = LaplaceSolver(num_pts=32)
    boundary_list = torch.tensor(
        [[0.5,0.5,0.25]]
        # [[0.25,0.25,0.125],
        #  [0.25,0.75,0.125],
        #  [0.75,0.25,0.125],
        #  [0.75,0.75,0.125]]
        )
    mask = solver.find_boundary(boundary_list)
    plt.imshow(mask,origin='lower')
    plt.show()

    solution = solver.solve(boundary_list, w=1.5, max_iter=100)
    plt.imshow(solution.squeeze(),origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()
    plt.imshow(solver.get_residual(solution,mask,sum=False),origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()


