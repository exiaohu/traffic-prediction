from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import init, functional as F


class SAdaptor(nn.Module):
    def __init__(self, supports: Tensor, node_dim: int, dropout: float):
        super(SAdaptor, self).__init__()
        u, s, v = torch.svd(supports)
        self.usv = u[..., :node_dim], s[..., :node_dim], v[..., :node_dim]
        self.bias = nn.Parameter(torch.empty_like(s[..., :node_dim]))
        init.ones_(self.bias)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self):
        u, s, v = self.usv
        s = s * F.softplus(self.bias)

        a = u.matmul(torch.diag_embed(s)).matmul(v.transpose(-1, -2))
        return self.dropout(a)


class TAdaptor(nn.Module):
    def __init__(self, n_hist: int, dropout: float):
        super(TAdaptor, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        # self.nodes = nn.Linear(n_hist, 16)

    def forward(self, inputs: Tensor):
        x = inputs[..., 0].transpose(1, 2)  # [B, N, T]
        # nodes = self.nodes(x)
        # nodes = (nodes - nodes.mean(dim=1, keepdim=True)) / nodes.std(dim=1, keepdim=True)
        nodes = F.normalize(x, p=2, dim=-1)

        src, dst = nodes.unsqueeze(1), nodes.unsqueeze(2)

        # d[torch.isinf(d) | torch.isnan(d)] = 0.0
        return self.dropout(torch.cdist(src, dst).squeeze(-1).neg().exp())


class STAdaptor(nn.Module):
    def __init__(self, supports: Tensor, n_hist: int, node_dim: int, dropout: float, spatial: bool, temporal: bool):
        super(STAdaptor, self).__init__()
        self.supports = supports
        self.supports.requires_grad_(False)

        if spatial:
            self.s_adaptor = SAdaptor(supports, node_dim, dropout)
        if temporal:
            self.t_adaptor = TAdaptor(n_hist, dropout)

    def forward(self, inputs: Optional[Tensor] = None) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        supports = self.supports

        if hasattr(self, 's_adaptor'):
            adaptive = self.s_adaptor()  # [1, N, N]
            # supports = torch.cat([supports, adaptive], dim=0)
            supports = supports + adaptive

        if hasattr(self, 't_adaptor'):
            dynamic = self.t_adaptor(inputs).unsqueeze(0)  # [1, B, N, N]
            supports = supports.unsqueeze(1)  # .expand(-1, dynamic.size(1), -1, -1)
            # supports = torch.cat([supports, dynamic], dim=0)
            supports = supports * (dynamic + 1)

        return F.normalize(torch.relu(supports), p=1, dim=-1)


def test_adaptor():
    s = torch.randn(2, 207, 207)
    m = STAdaptor(s, 12, 16, 0.5, False, False)
    x = torch.randn(64, 12, 207, 2)
    print(m().shape)
    m = STAdaptor(s, 12, 16, 0.5, True, False)
    print(m().shape)
    m = STAdaptor(s, 12, 16, 0.5, False, True)
    print(m(x).shape)
    m = STAdaptor(s, 12, 16, 0.5, True, True)
    print(m(x).shape)
