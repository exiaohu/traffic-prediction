import torch
from torch import nn, Tensor
from torch.nn import functional as F


class TAdaptor(nn.Module):
    def __init__(self, n_hist: int, n_in: int, node_dim: int, dropout: float):
        super(TAdaptor, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.nodes = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 2 * node_dim, kernel_size=(1, n_hist))
        )
        # self.ebd = nn.Parameter(torch.empty(nodes_num, node_dim))
        # init.xavier_normal_(self.ebd, math.sqrt(2))

    def forward(self, inputs: Tensor, supports: Tensor):
        """
        :param inputs: tensor, [B, T, N, F]
        :param supports: tensor, [E, N, N]
        :return: tensor, [E, B, N, N]
        """
        x = inputs.transpose(1, 3)
        nodes = self.nodes(x).squeeze(3).transpose(1, 2)  # [B, N, D]

        self.dropout(nodes)

        b, n, dd = nodes.shape
        w, b = torch.split(nodes, dd // 2, dim=2)
        w = torch.einsum('bud,bvd->buv', [w, w])  # [B, N, N]
        b = torch.einsum('bud,bvd->buv', [b, b])  # [B, N, N]

        return supports.unsqueeze(1) * (w + 1) + b


class STAdaptor(nn.Module):
    def __init__(self, supports: Tensor, n_hist: int, node_dim: int, dropout: float, spatial: bool, temporal: bool):
        super(STAdaptor, self).__init__()
        self.adaptive = nn.Parameter(supports, requires_grad=spatial)

        if temporal:
            self.t_adaptor = TAdaptor(n_hist, 9, node_dim, dropout)

    def forward(self, inputs: Tensor = None) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        supports = self.adaptive

        if hasattr(self, 't_adaptor'):
            supports = self.t_adaptor(inputs, supports)

        return F.normalize(torch.relu(supports), p=1, dim=-1)


def test_adaptor():
    s = torch.randn(2, 207, 207)
    m = STAdaptor(s, 12, 16, 0.5, False, False)
    x = torch.randn(64, 12, 207, 9)
    print(m(x).shape)
    m = STAdaptor(s, 12, 16, 0.5, True, False)
    print(m(x).shape)
    m = STAdaptor(s, 12, 16, 0.5, False, True)
    print(m(x).shape)
    m = STAdaptor(s, 12, 16, 0.5, True, True)
    print(m(x).shape)
