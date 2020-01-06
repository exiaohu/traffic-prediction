from typing import Optional

import torch
from torch import nn, Tensor


class SAdaptor(nn.Module):
    def __init__(self, nodes_num: int, node_dim: int, edge_dim: int, dropout: float):
        super(SAdaptor, self).__init__()
        self.nodes = nn.Embedding(nodes_num, node_dim)
        self.mapping = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, edge_dim)
        )

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, supports: Tensor):
        idx = torch.tensor(range(self.nodes.num_embeddings), dtype=torch.long, device=supports.device)
        nodes = self.nodes(idx)
        mapping = torch.abs(nodes.unsqueeze(0) - nodes.unsqueeze(1))  # [N, N, node_dim]
        mapping = self.mapping(mapping)  # [N, N, edge_dim]

        mapping = (mapping - mapping.mean(dim=[0, 1], keepdim=True)) / mapping.std(dim=[0, 1], keepdim=True)
        mapping = mapping.permute(2, 0, 1)

        self.dropout(mapping)

        return supports * (mapping + 1)


# class TAdaptor(nn.Module):
#     def __init__(self, in_dim: int, node_dim: int, edge_dim: int, dropout: float):
#         super(TAdaptor, self).__init__()
#
#         self.convert = nn.Sequential(
#             nn.Conv2d(in_dim, 64, kernel_size=(1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=(1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(64, 2 * node_dim, kernel_size=(1, 1))
#         )
#
#         self.mapping = nn.Sequential(
#             nn.Conv2d(node_dim + edge_dim, 64, kernel_size=(1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=(1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(64, edge_dim, kernel_size=(1, 1)),
#         )
#
#         self.dropout = nn.Dropout(dropout, inplace=True)
#
#     def forward(self, supports: Tensor, inputs: Tensor):
#         vals = self.convert(inputs.transpose(1, 3))  # [B, 2 * node_dim, N, T]
#         gate, vals = torch.split(vals, vals.size(1) // 2, dim=1)
#         vals = torch.sum(torch.sigmoid(gate) * torch.tanh(vals), dim=-1)  # [B, node_dim, N]
#         vals = vals.transpose(1, 2)  # [B, N, node_dim]
#         vals = vals.unsqueeze(1) + vals.unsqueeze(2)  # [B, N, N, node_dim]
#
#         vals = vals.permute(3, 0, 1, 2)
#         _, b, _, _ = vals.shape
#         supports = supports.unsqueeze(1).expand(-1, b, -1, -1)
#         vals = torch.cat([supports, vals]).transpose(0, 1)
#
#         return self.mapping(vals).transpose(0, 1) + supports


class TAdaptor(nn.Module):
    def __init__(self, n_hist: int):
        super(TAdaptor, self).__init__()
        self.w = nn.MultiheadAttention(n_hist, 3)
        self.b = nn.MultiheadAttention(n_hist, 3)

    def forward(self, supports: Tensor, inputs: Tensor):
        x = inputs[..., 0].permute(2, 0, 1)
        _, w = self.w(x, x, x)
        _, b = self.b(x, x, x)
        supports = supports.unsqueeze(1)

        return supports * w + b


class STAdaptor(nn.Module):
    def __init__(self, supports: Tensor, n_hist: int, node_dim: int, dropout: float):
        super(STAdaptor, self).__init__()
        self.supports = supports
        self.supports.requires_grad_(False)

        edge_dim, nodes_num, _ = supports.shape

        self.s_adaptor = SAdaptor(nodes_num, node_dim, edge_dim, dropout)
        self.t_adaptor = TAdaptor(n_hist)

    def forward(self, spatial: bool, temporal: bool, inputs: Optional[Tensor] = None) -> Tensor:
        """
        :param spatial: bool
        :param temporal: bool
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        supports = self.supports

        if spatial:
            supports = supports + self.s_adaptor(supports)

        if temporal:
            supports = supports.unsqueeze(1) + self.t_adaptor(supports, inputs)

        return self.normalize(torch.relu(supports))

    @staticmethod
    def normalize(adjs: Tensor):
        """
        :param adjs: tensor, [E, N, N] or [E, B, N, N]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        d_inv_sqrt = adjs.sum(-1).pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt_left, d_inv_sqrt_right = d_inv_sqrt.unsqueeze(-1), d_inv_sqrt.unsqueeze(-2)
        return d_inv_sqrt_left * adjs * d_inv_sqrt_right


def test_adaptor():
    s = torch.randn(2, 207, 207)
    m = STAdaptor(s, 12, 16, 0.5)
    x = torch.randn(64, 12, 207, 2)
    print(m(False, False).shape)
    print(m(True, False).shape)
    print(m(False, True, x).shape)
    print(m(True, True, x).shape)
