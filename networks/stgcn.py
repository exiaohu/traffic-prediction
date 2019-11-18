from typing import Tuple, List

import torch
from torch import nn, Tensor

from .base import CausalConv1d, ChebNet


class ResShortcut(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        super(ResShortcut, self).__init__()
        if f_in > f_out:
            self.alignment = nn.Linear(f_in, f_out)
        self.f_in, self.f_out = f_in, f_out

    def forward(self, inputs):
        # residual connection, first map the input to the same shape as output
        if self.f_in > self.f_out:
            return self.alignment(inputs)
        elif self.f_in < self.f_out:
            zero_shape = inputs.shape[:-1] + (self.f_out - self.f_in,)
            zeros = torch.zeros(zero_shape, dtype=inputs.dtype, device=inputs.device)
            return torch.cat([inputs, zeros], dim=-1)
        return inputs


class TemporalConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, kernel_size: int):
        super(TemporalConvLayer, self).__init__()
        self.causal_conv = CausalConv1d(f_in, 2 * f_out, kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.shortcut = ResShortcut(f_in, f_out)

        self.f_in, self.f_out, self.kernel_size = f_in, f_out, kernel_size

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Temporal causal convolution layer.
        :param inputs: tensor, [B, T, N, F_in]
        :return: tensor, [B, T - kernel_size + 1, N, F_out]
        """
        b, t, n, _ = inputs.shape

        x_res = self.shortcut(inputs[:, self.kernel_size - 1:, :, :])

        # shape => [B * N, F_in, T]
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, self.f_in, t)

        # equal shape => [B * N, 2 * F_out, T - kernel_size + 1] => [B, T - kernel_size + 1, N, 2 * F_out]
        outputs = self.causal_conv(inputs).reshape(b, n, 2 * self.f_out, -1).permute(0, 3, 1, 2)

        return (outputs[..., :self.f_out] + x_res) * self.sigmoid(outputs[..., -self.f_out:])


class SpatialConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, k_hop: int):
        super(SpatialConvLayer, self).__init__()
        self.g_conv = ChebNet(f_in, f_out, k_hop)
        self.relu = nn.ReLU()
        self.shortcut = ResShortcut(f_in, f_out)

        self.f_in, self.f_out = f_in, f_out

    def forward(self, inputs: Tensor, cheb_filters: Tensor) -> Tensor:
        """
        Spatial graph convolution layer.
        :param inputs: tensor, [B, T, N, F_in]
        :param cheb_filters: tensor, [N, K_hop, N]
        :return: tensor, [B, T, N, F_out]
        """
        x_res = self.shortcut(inputs)
        outputs = self.g_conv(inputs, cheb_filters)
        return self.relu(outputs + x_res)


class STConvBlock(nn.Module):
    def __init__(self,
                 k_hop: int,
                 t_cnv_krnl_sz: int,
                 n_node: int,
                 channels: Tuple[int, int, int],
                 dropout: float):
        """
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param k_hop: length of Chebychev polynomial, i.e., kernel size of spatial convolution
        :param t_cnv_krnl_sz: kernel size of temporal convolution
        :param n_node: the number of nodes
        :param channels: three integers, define each of the sub-blocks
        :param dropout: dropout
        """
        super(STConvBlock, self).__init__()
        f_in, f_m, f_out = channels
        self.t_conv1 = TemporalConvLayer(f_in, f_m, t_cnv_krnl_sz)
        self.s_conv = SpatialConvLayer(f_m, f_m, k_hop)
        self.t_conv2 = TemporalConvLayer(f_m, f_out, t_cnv_krnl_sz)
        self.ln = nn.LayerNorm([n_node, f_out])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, cheb_filters: Tensor) -> Tensor:
        """
        forward of spatio-temporal convolutional block
        :param inputs: tensor, [B, T, N, F_in]
        :param cheb_filters: tensor, [N, K_hop, N]
        :return: tensor, [B, T, N, F_out]
        """
        outputs = self.t_conv1(inputs)
        outputs = self.s_conv(outputs, cheb_filters)
        outputs = self.t_conv2(outputs)
        return self.dropout(self.ln(outputs))


class OutputLayer(nn.Module):
    def __init__(self, f_in: int, t_cnv_krnl_sz: int, n_node: int):
        super(OutputLayer, self).__init__()
        self.t_conv1 = TemporalConvLayer(f_in, f_in, t_cnv_krnl_sz)
        self.ln = nn.LayerNorm([n_node, f_in])
        self.t_conv2 = TemporalConvLayer(f_in, f_in, 1)
        self.out = nn.Sequential(nn.Linear(f_in, 1), nn.ReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Output layer: temporal convolution layers attach with one fully connected layer,
        which map outputs of the last st_conv block to a single-step prediction.
        :param inputs: tensor, [B, 1, N, F]
        :return: tensor, [B, N, 1]
        """
        outputs = self.t_conv1(inputs)
        outputs = self.ln(outputs)
        outputs = self.t_conv2(outputs)
        return self.out(outputs).squeeze(1)


class STGCN(nn.Module):
    def __init__(self,
                 n_history: int,
                 k_hop: int,
                 t_cnv_krnl_sz: int,
                 n_node: int,
                 dims: List[Tuple[int, int, int]],
                 dropout: float):
        super(STGCN, self).__init__()
        self.st_blocks = nn.ModuleList([STConvBlock(k_hop, t_cnv_krnl_sz, n_node, dim, dropout) for dim in dims])

        n_history -= 2 * (t_cnv_krnl_sz - 1) * len(dims)

        self.out = OutputLayer(dims[-1][-1], n_history, n_node)

    def forward(self, inputs: Tensor, cheb_filters: Tensor) -> Tensor:
        """
        STGCN product single step prediction
        :param inputs: tensor, [B, T, N, F]
        :param cheb_filters: [N, K_hop, N]
        :return: tensor, [B, N, 1]
        """
        outputs = inputs
        for _, st_block in enumerate(self.st_blocks):
            outputs = st_block(outputs, cheb_filters)
        return self.out(outputs)
