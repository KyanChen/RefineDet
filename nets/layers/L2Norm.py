import torch


class L2Norm(torch.nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.gamma = scale or None
        self.eps = 1e-10
        self.weights = torch.nn.Parameter(torch.Tensor(self.n_channels))
        torch.nn.init.constant_(self.weights, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

