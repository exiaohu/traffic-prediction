import math
from typing import List, Callable

import scipy.sparse as sp
import torch
from torch import nn, Tensor
from torch.nn import init


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, edge_dim: int):
        super(GraphConv, self).__init__()
        self.c_in, self.c_out, self.edge_dim = c_in, c_out, edge_dim

        c_in = (edge_dim + 1) * c_in
        self.out_w = nn.Parameter(torch.empty(c_in, c_out))
        self.out_b = nn.Parameter(torch.empty(c_out))
        init.xavier_normal_(self.out_w, gain=math.sqrt(2))
        init.constant_(self.out_b, 1.0)

    def forward(self, x: Tensor, supports: Tensor):
        """
        :param x: tensor, [B, c_in, N, T] or [B, c_in, N]
        :param supports: tensor, [n_edge, N, N] or [n_edge, B, N, N]
        :return: tensor, [B, c_out, N, T] or [B, c_out, N]
        """
        h = [x] + [self.nconv(x, a) for a in supports]
        h = torch.cat(h, 1)
        return self.out(h)

    @staticmethod
    def nconv(x: Tensor, a: Tensor):
        assert len(a.shape) in [2, 3] and len(x.shape) in [3, 4], f'x of {x.shape} or adj of {a.shape} is wrong.'
        x_, r_ = ('bfvc', 'bfwc') if len(x.shape) == 4 else ('bfv', 'bfw')
        a_ = 'vw' if len(a.shape) == 2 else 'bvw'
        x = torch.einsum(f'{x_},{a_}->{r_}', [x, a])
        return x.contiguous()

    def out(self, h: Tensor):
        w, b = self.out_w, self.out_b
        c_in, c_out = w.shape

        b = b.reshape(c_out, 1, 1) if len(h.shape) == 4 else b.reshape(c_out, 1)
        h_, r_ = ('biwc', 'bowc') if len(h.shape) == 4 else ('biw', 'bow')
        return torch.einsum(f'{h_},io->{r_}', [h, w]) + b

    def __repr__(self):
        return f'{self._get_name()}(c_in={self.c_in}, c_out={self.c_out}, edge_dim={self.edge_dim})'


class StackedGraphConv(nn.ModuleList):
    def __init__(self, dims: List[int], activation: Callable, edge_dim: int):
        super(StackedGraphConv, self).__init__()
        self.activation = activation
        for cin, cout in zip(dims[:-1], dims[1:]):
            self.append(GraphConv(cin, cout, edge_dim))

    def forward(self, x: Tensor, supports):
        h = x
        for graph_conv in self:
            h = graph_conv(self.activation(h), supports)
        return h


class SelfAttention(nn.Module):
    def __init__(self, n_hidden: int, dim: int, in_len: int = 5):
        super(SelfAttention, self).__init__()
        self.dim = dim
        if in_len == 5:
            self.projector = nn.Sequential(
                nn.Conv3d(n_hidden, 64, kernel_size=(1, 1, 1)),
                nn.ReLU(True),
                nn.Conv3d(64, 64, kernel_size=(1, 1, 1)),
                nn.ReLU(True),
                nn.Conv3d(64, 1, kernel_size=(1, 1, 1)),
            )
        elif in_len == 4:
            self.projector = nn.Sequential(
                nn.Conv2d(n_hidden, 64, kernel_size=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(64, 1, kernel_size=(1, 1)),
            )

    def forward(self, inputs: Tensor):
        """
        :param inputs: tensor, [B, C, X, N, T]
        :return: tensor, [B, C, N, T] or [B, C, N]
        """
        energy = self.projector(inputs)

        weights = energy
        weights = torch.softmax(weights, dim=self.dim)
        outputs = (inputs * weights).sum(self.dim)
        return outputs


class STAdaptor(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, node_dim: int, nodes_num: int, dynamic: bool):
        super(STAdaptor, self).__init__()
        self.edge_dim = edge_dim

        self.node_embedding = nn.Parameter(torch.empty(nodes_num, node_dim))
        init.xavier_normal_(self.node_embedding, math.sqrt(2))

        self.adaptor_w = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, edge_dim)
        )

        if dynamic:
            self.dynamic_convert = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_dim, 64, kernel_size=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(64, node_dim, kernel_size=(1, 1)),
                nn.ReLU(True),
                SelfAttention(node_dim, dim=3, in_len=4)
            )

    def forward(self, supports: Tensor, inputs: Tensor) -> Tensor:
        """
        :param supports: tensor, [E, N, N]
        :param inputs: tensor, [B, C, N, T]
        :return: tensor, [E, N, N]
        """
        supports = supports.permute(1, 2, 0)  # [N, N, E]

        ne = self.node_embedding

        if hasattr(self, 'dynamic_convert'):
            dynamic_component = self.dynamic_convert(inputs)
            dynamic_component = torch.transpose(dynamic_component, 1, 2)
            ne = ne + dynamic_component

        ne = ne.unsqueeze(-3) + ne.unsqueeze(-2)
        adaptive = self.adaptor_w(ne)

        adaptive = (adaptive - torch.mean(adaptive)) / torch.std(adaptive)
        adaptive = (adaptive + 1) * supports
        adaptive = adaptive.unsqueeze(0).transpose(0, -1).squeeze(-1)
        adaptive = torch.relu(adaptive)

        return self.normalize(adaptive)

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


class STLayer(nn.ModuleList):
    def __init__(self, n_hidden: int, edge_dim: int, dropout: float):
        super(STLayer, self).__init__()
        self.edge_dim = edge_dim

        self.bn = nn.BatchNorm2d(n_hidden)

        self.residual = nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1))

        self.affine_conv = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 2))
        )

        self.gate_conv = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 2))
        )

        self.spatial_conv = GraphConv(n_hidden, n_hidden, edge_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, supports):
        # supports = self.supports(x)
        # residual = self.residual(x)
        x = torch.relu(self.bn(x))

        residual = self.residual(x)

        affine = self.affine_conv(x)
        gate = self.gate_conv(x)
        x = torch.tanh(affine + residual) * torch.sigmoid(gate)
        x = self.spatial_conv(x, supports)

        return self.dropout(x)


class DenseBlock(nn.Module):
    def __init__(self, n_layers: int, n_hid, e_dim: int, dropout: float):
        super(DenseBlock, self).__init__()
        self.n_hiddens = n_hid

        self.layers = nn.ModuleList([STLayer(n_hid, e_dim, dropout) for _ in range(n_layers)])
        self.out = SelfAttention(n_hid, dim=2)
        # self.out = nn.Conv2d(n_hid * (n_layers + 1), n_hid, kernel_size=(1, 1))

    def forward(self, x: Tensor, supports) -> Tensor:
        b, _, n, t = x.shape
        h = x
        x = x.unsqueeze(2)
        for layer in self.layers:
            h = layer(h, supports)
            x = torch.cat([x, h.unsqueeze(2)], 2)
        # x = x.reshape(b, -1, n, t)
        return self.out(torch.relu(x))


class Ours(nn.Module):
    def __init__(self,
                 factor,
                 n_in: int,
                 n_out: int,
                 n_hidden: int,
                 t_pred: int,
                 n_blocks: int,
                 n_layers: int,
                 edge_dim: int,
                 node_dim: int,
                 nodes_num: int,
                 dropout: float,
                 dynamic: bool):
        super(Ours, self).__init__()
        self.t_pred = t_pred
        # n_in = n_in + 2

        self.support_adaptor = STAdaptor(n_hidden, edge_dim, node_dim, nodes_num, dynamic)
        self.raw_supports = factor

        self.enter = nn.Conv2d(n_in, n_hidden, kernel_size=(1, 1))

        self.blocks = nn.ModuleList(
            [DenseBlock(n_layers, n_hidden, edge_dim, dropout) for _ in range(n_blocks)]
        )

        self.temporal_reduce = SelfAttention(n_hidden, dim=4)

        self.out = StackedGraphConv([n_hidden * (n_blocks + 1), 128, 256, t_pred * n_out], torch.relu, edge_dim)

    def forward(self, inputs: Tensor):
        """
        : params inputs: tensor, [B, T, N, F]
        """
        inputs = inputs.transpose(1, 3)  # [B, F, N, T]

        x = self.enter(inputs)

        # supports = self.raw_supports
        supports = self.support_adaptor(self.raw_supports, x)

        b, c, n, t = x.shape

        h = x
        x = x.unsqueeze(2)
        for block in self.blocks:
            h = block(h, supports)
            x = torch.cat([x, h.unsqueeze(2)], 2)
        h = self.temporal_reduce(torch.relu(x))
        h = torch.reshape(h, (b, -1, n))
        y_ = self.out(h, supports)

        return y_.reshape(b, self.t_pred, -1, n).transpose(-1, -2)


def test_ours():
    from utils.utils import get_number_of_parameters
    from utils import load_graph_data
    supports = load_graph_data('METR-LA', 'doubletransition')
    supports = torch.tensor(list(map(sp.coo_matrix.toarray, supports)), dtype=torch.float32)
    config = {
        'n_in': 2,
        'n_out': 1,
        't_pred': 12,
        'n_hidden': 32,
        'n_blocks': 4,
        'n_layers': 3,
        'edge_dim': 2,
        'node_dim': 8,
        'nodes_num': 207,
        'dropout': 0.3,
        'dynamic': False
    }
    batch_size = 64
    m = Ours(supports, **config)
    print(m)
    print(get_number_of_parameters(m))
    x = torch.randn(batch_size, 12, config['nodes_num'], config['n_in'])
    y = m(x)
    print(y.shape)


def test_g_conv():
    m = GraphConv(32, 32, 4)
    x = torch.randn(64, 32, 207, 12)
    s = torch.randn(4, 207, 207)
    y = m(x, s)
    print(y.shape)
