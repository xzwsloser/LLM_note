import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(size=(dim,)))
        self.eps = eps
    def forward(self, X):
       # X.shape -> (batch_size, dim)
       p2 = X.square().sum(dim=-1,keepdim=True)
       div = torch.sqrt(p2/self.dim + self.eps)
       X = X / div * self.gamma
       return X
if __name__ == '__main__':
    norm = RMSNorm(2)
    X = torch.tensor([
        [1, 2],
        [3, 4]
    ])
    X = norm(X)
    print(X)
