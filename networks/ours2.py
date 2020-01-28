from typing import List, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, edge_dim: int):
        super(GraphConv, self).__init__()
        self.c_in, self.c_out, self.edge_dim = c_in, c_out, edge_dim
        self.out = nn.Conv2d(c_in * (edge_dim + 1), c_out, kernel_size=(1, 1))

    def forward(self, x: Tensor, supports: Tensor):
        """
        :param x: tensor, [B, c_in, N, T] or [B, c_in, N]
        :param supports: tensor, [n_edge, N, N] or [n_edge, B, N, N]
        :return: tensor, [B, c_out, N, T] or [B, c_out, N]
        """
        flag = (len(x.shape) == 3)
        if flag:
            x.unsqueeze_(-1)

        h = [x] + [self.nconv(x, a) for a in supports]
        h = torch.cat(h, 1)
        return self.out(h).squeeze(-1) if flag else self.out(h)

    @staticmethod
    def nconv(x: Tensor, a: Tensor):
        """
        :param x: tensor, [B, C, N, T]
        :param a: tensor, [B, N, N] or [N, N]
        :return:
        """
        a_ = 'vw' if len(a.shape) == 2 else 'bvw'
        x = torch.einsum(f'bcvt,{a_}->bcwt', [x, a])
        return x.contiguous()

    def __repr__(self):
        return f'GraphConv({self.c_in}, {self.c_out}, edge_dim={self.edge_dim}, attn={hasattr(self, "attn")})'


class StackedGraphConv(nn.ModuleList):
    def __init__(self, dims: List[int], act: Callable, edge_dim: int, dropout: float):
        super(StackedGraphConv, self).__init__(
            [GraphConv(c_in, c_out, edge_dim) for c_in, c_out in zip(dims[:-1], dims[1:])])
        self.act = act
        self.dropout = dropout

    def forward(self, x: Tensor, supports):
        h = x
        for graph_conv in self:
            h = torch.dropout(h, p=self.dropout, train=self.training)
            h = graph_conv(self.act(h), supports)
        return h


class SelfAttention(nn.Module):
    def __init__(self, n_in: int, dims: List[int], n_out: int):
        super(SelfAttention, self).__init__()
        modules = list()
        for i, (c_in, c_out) in enumerate(zip([n_in] + dims, dims + [n_out])):
            modules.append(nn.Conv3d(c_in, c_out, kernel_size=(1, 1, 1)))
            if i != len(dims):
                modules.append(nn.ReLU(True))
        self.projector = nn.Sequential(*modules)

    def forward(self, inputs: Tensor, reduce_dim: int):
        """
        :param inputs: tensor, [B, C, X, Y, Z]
        :param reduce_dim: int
        :return: tensor, [B, C, ...]
        """
        energy = self.projector(inputs)

        inputs = inputs.unsqueeze(1)  # [B, 1,  C_in, X, Y, Z]
        energy = energy.unsqueeze(2)  # [B, C_out, 1, X, Y, Z]

        weights = torch.softmax(energy, dim=reduce_dim + 1)
        outputs = (inputs * weights).sum(reduce_dim + 1)
        return outputs.squeeze(1)


class STLayer(nn.ModuleList):
    def __init__(self, n_hid: int, edge_dim: int, t_size: int, i_layer: int, dropout: float):
        super(STLayer, self).__init__()
        self.edge_dim = edge_dim
        self.bn = nn.BatchNorm2d(n_hid)

        # stride, dilation = t_size ** (i_layer + 1), t_size ** i_layer
        stride, dilation = 1, 1

        self.residual = nn.Conv2d(n_hid, n_hid, kernel_size=(1, 1))
        self.affine_conv = nn.Conv2d(n_hid, n_hid, kernel_size=(1, t_size), stride=(1, stride), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(n_hid, n_hid, kernel_size=(1, t_size), stride=(1, stride), dilation=(1, dilation))

        self.spatial_conv = GraphConv(n_hid, n_hid, edge_dim)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def pad_inside(x: Tensor, stride: int) -> Tensor:
        shape = (1, stride)
        w = x.new_zeros(shape)
        w[0, 0] = 1.0
        return F.conv_transpose2d(x, w.expand(x.size(1), 1, *shape), stride=shape, groups=x.size(1))

    def forward(self, x: Tensor, supports):
        stride = max(self.affine_conv.stride[-1], self.gate_conv.stride[-1])
        assert stride <= x.size(-1), 't_size is too big or neural network is too deep.'
        residual = self.residual(x)
        x = torch.relu(self.bn(x))

        affine = self.affine_conv(x)
        gate = self.gate_conv(x)
        if stride > 1:
            affine = self.pad_inside(affine, stride)
            gate = self.pad_inside(gate, stride)

        if affine.size(-1) < x.size(-1):
            affine = F.pad(affine, [0, x.size(-1) - affine.size(-1), 0, 0])
        if gate.size(-1) < x.size(-1):
            gate = F.pad(gate, [0, x.size(-1) - gate.size(-1), 0, 0])

        x = torch.tanh(affine + residual) * torch.sigmoid(gate)

        x = self.spatial_conv(x, supports)

        return self.dropout(x)


class DenseBlock(nn.Module):
    def __init__(self, n_layers: int, n_hid, e_dim: int, t_size: int, dropout: float):
        super(DenseBlock, self).__init__()
        self.n_hiddens = n_hid

        self.layers = nn.ModuleList([STLayer(n_hid, e_dim, t_size, i, dropout) for i in range(n_layers)])
        self.out = SelfAttention(n_hid, [64, 64], 1)

    def forward(self, x: Tensor, supports) -> Tensor:
        h = x
        x = x.unsqueeze(2)
        for layer in self.layers:
            h = layer(h, supports)
            x = torch.cat([x, h.unsqueeze(2)], 2)

        return self.out(torch.relu(x), 2)


class Ours(nn.Module):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 n_hidden: int,
                 t_pred: int,
                 n_layers: List[int],
                 t_sizes: List[int],
                 edge_dim: int,
                 expand_dims: List[int],
                 dropout: float):
        super(Ours, self).__init__()
        self.t_pred = t_pred
        self.n_out = n_out

        self.enter = nn.Conv2d(n_in, n_hidden, kernel_size=(1, 1))

        self.blocks = nn.ModuleList(
            [DenseBlock(n_layer, n_hidden, edge_dim, tsize, dropout) for n_layer, tsize in zip(n_layers, t_sizes)]
        )

        self.temporal_reduce = SelfAttention(n_hidden, [64, 64], 1)
        out_dim = n_hidden * (len(n_layers) + 1)
        self.out = StackedGraphConv([out_dim, *expand_dims, t_pred * n_out], torch.relu, edge_dim, dropout)

    def forward(self, inputs: Tensor, supports: Tensor):
        """
        :param inputs: tensor, [B, T, N, F]
        :param supports: tensor, [E, B, N, N] or [E, N, N]
        :return
        """
        inputs = inputs.transpose(1, 3)  # [B, F, N, T]

        x = self.enter(inputs)
        b, c, n, t = x.shape

        h = x
        x = x.unsqueeze(2)
        for block in self.blocks:
            h = block(h, supports)
            x = torch.cat([x, h.unsqueeze(2)], 2)

        h = self.temporal_reduce(torch.relu(x), 4)  # [B, F, L, N]
        h = h.reshape(b, -1, n, 1)  # [B, F, N, 1]
        y_ = self.out(h, supports)  # [B, n_pred * n_out, N, 1]

        return y_.reshape(b, self.t_pred, -1, n).transpose(2, 3)

    def produce(self, go_symbol: Tensor, cands: Tensor):
        """
        :param go_symbol: tensor, [B, C, N, 1]
        :param cands: tensor, [B, C, N, X]
        :return:
        """
        b, c, n, _ = go_symbol.shape
        go_symbol = go_symbol.permute(3, 0, 2, 1).reshape(-1, b * n, c)
        cands = cands.permute(3, 0, 2, 1).reshape(-1, b * n, c)

        x, y = go_symbol, torch.empty(0, b * n, c, device=go_symbol.device)
        for _ in range(self.t_pred):
            x_l, _ = self.attn(x, cands, cands)
            x_s, _ = self.attn2(x, y, y)
            gate = torch.sigmoid(self.gate(x_l))
            x = gate * x_l + torch.rsub(gate, 1.0) * x_s
            y = torch.cat([y, x], dim=0)

        return y.reshape(-1, b, n, c).transpose(0, 1)


def test_ours():
    from utils.utils import get_number_of_parameters
    from utils import load_graph_data
    import scipy.sparse as sp
    supports = load_graph_data('METR-LA', 'doubletransition')
    supports = torch.tensor(list(map(sp.coo_matrix.toarray, supports)), dtype=torch.float32)
    config = {
        'n_in': 2,
        'n_out': 1,
        't_pred': 12,
        'n_hidden': 32,
        'n_layers': [3, 3, 3, 3],
        'edge_dim': 2,
        't_sizes': [2, 2, 2, 2],
        'expand_dims': [256],
        'dropout': 0.3,
    }
    batch_size = 64
    m = Ours(**config)
    print(m)
    print(get_number_of_parameters(m))
    x = torch.randn(batch_size, 12, 207, config['n_in'])
    y = m(x, supports)
    print(y.shape)


def test_g_conv():
    m = GraphConv(32, 32, 4)
    x = torch.randn(64, 32, 207, 12)
    s = torch.randn(4, 207, 207)
    print(m(x, s).shape)
    m = GraphConv(32, 32, 4)
    print(m(x, s).shape)
