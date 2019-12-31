from typing import Tuple, Union

import dgl
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from networks.ours2 import GraphConv


class STLayer(nn.Module):
    def __init__(self, n_residuals: int, n_dilations: int, kernel_size: int, dilation: int,
                 n_skip: int, edge_dim: int):
        super(STLayer, self).__init__()
        # dilated convolutions
        self.filter_conv = nn.Conv2d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)

        self.gate_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)

        # 1x1 convolution for residual connection
        self.gconv = GraphConv(n_dilations, n_residuals, edge_dim)

        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(n_dilations, n_skip, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(n_residuals)

    def forward(self, x: Tensor, skip: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        residual = x
        # dilated convolution
        _filter = self.filter_conv(residual)
        _filter = torch.tanh(_filter)
        _gate = self.gate_conv(residual)
        _gate = torch.sigmoid(_gate)
        x = _filter * _gate

        # parametrized skip connection
        s = x
        s = self.skip_conv(s)
        skip = skip[:, :, :, -s.size(3):]
        skip = s + skip

        x = self.gconv(x, supports)

        x = x + residual[:, :, :, -x.size(3):]

        x = self.bn(x)
        return x, skip


class STBlock(nn.ModuleList):
    def __init__(self, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, edge_dim: int):
        super(STBlock, self).__init__()
        for i in range(n_layers):
            self.append(
                STLayer(n_residuals, n_dilations, kernel_size, 2 ** i, n_skips, edge_dim))

    def forward(self, x: Tensor, skip: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        for layer in self:
            x, skip = layer(x, skip, supports)

        return x, skip


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, n_blocks, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, edge_dim: int):
        self.n_skips = n_skips
        super(StackedSTBlocks, self).__init__()
        for _ in range(n_blocks):
            self.append(
                STBlock(n_layers, kernel_size, n_residuals, n_dilations, n_skips, edge_dim))

    def forward(self, x: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        b, f, n, t = x.shape
        skip = torch.zeros(b, self.n_skips, n, t, dtype=torch.float32, device=x.device)
        for block in self:
            x, skip = block(x, skip, supports)
        return x, skip


class Ours(nn.Module):
    def __init__(self,
                 factor: Tensor,
                 n_in: int,
                 n_out: int,
                 n_pred: int,
                 edge_dim: int,
                 n_residuals: int,
                 n_dilations: int,
                 n_skips: int,
                 n_ends: int,
                 kernel_size: int,
                 n_blocks: int,
                 n_layers: int):
        super(Ours, self).__init__()
        # n_in = n_in + 2
        self.t_pred = n_pred

        self.factor = factor

        self.receptive_field = n_blocks * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        self.enter = nn.Conv2d(n_in, n_residuals, kernel_size=(1, 1))

        self.blocks = StackedSTBlocks(n_blocks, n_layers, kernel_size, n_residuals, n_dilations,
                                      n_skips, edge_dim)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_skips, n_ends, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_ends, n_pred * n_out, kernel_size=(1, 1))
        )

    def forward(self, inputs: Tensor):
        """
        : params inputs: tensor, [B, T, N, F]
        """
        inputs = inputs.transpose(1, 3)
        # static_x, dynamic_x = self.inputs_split(inputs[:, 0, ...])
        # inputs = torch.cat([inputs, static_x.unsqueeze(1), dynamic_x.unsqueeze(1)], dim=1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = F.pad(inputs, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = inputs
        x = self.enter(x)

        b, c, n, t = x.shape

        x, skip = self.blocks(x, self.factor)

        y_ = self.out(skip)
        return y_.reshape(b, self.t_pred, -1, n).transpose(-1, -2)

    def inputs_split(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param data: tensor, shape [B, N, T]
        """
        static_x = torch.einsum('bnt,nu,up->bpt', [data, self.factor, self.factor.t()])
        return static_x, data - static_x


def test_ours():
    config = {
        'n_in': 2,
        'n_out': 1,
        'n_pred': 12,
        'edge_dim': 2,
        'n_residuals': 32,
        'n_dilations': 32,
        'n_skips': 256,
        'n_ends': 512,
        'kernel_size': 2,
        'n_blocks': 4,
        'n_layers': 2,
    }
    batch_size = 64
    from utils.utils import get_number_of_parameters
    # factor = node_embedding('METR-LA', 100)
    factor = torch.randn(2, 207, 207)
    m = Ours(factor, **config)

    print(get_number_of_parameters(m))
    x = torch.randn(batch_size, config['n_hist'], 207, config['n_in'])
    y = m(x)
    print(y.shape)
