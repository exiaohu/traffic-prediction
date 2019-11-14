from typing import Tuple, List

import torch
from torch import nn

from .base import CausalConv1d, ChebNet


class TemporalConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, kernel_size: int = 2):
        super(TemporalConvLayer, self).__init__()
        self.causal_conv = CausalConv1d(f_in, 2 * f_out, kernel_size)
        self.sigmoid = nn.Sigmoid()

        if f_in > f_out:
            self.alignement = nn.Linear(f_in, f_out)

        self.f_in, self.f_out, self.kernel_size = f_in, f_out, kernel_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Temporal causal convolution layer.
        :param inputs: tensor, [B, T, N, F_in]
        :return: tensor, [B, T - kernel_size + 1, N, F_out]
        """
        b, t, n, _ = inputs.shape

        # residual connection, first map the input to the same shape as output
        ori_input = inputs[:, self.kernel_size - 1:t, :, :]
        if self.f_in > self.f_out:
            ori_input = self.alignement(ori_input)
        elif self.f_in < self.f_out:
            zero_shape = ori_input.shape[:3] + (self.f_out - self.f_in,)
            ori_input = torch.cat([ori_input, torch.zeros(zero_shape, device=ori_input.device)], dim=3)

        # shape => [B * N, F_in, T]
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, self.f_in, t)

        # equal shape => [B * N, F_out, T - kernel_size + 1]
        p, q = self.causal_conv(inputs).split(self.f_out, dim=1)

        # reorder output and input to [B, T', N, F']
        p = p.reshape(b, n, self.f_out, -1).permute(0, 3, 1, 2)
        q = q.reshape(b, n, self.f_out, -1).permute(0, 3, 1, 2)

        return (p + ori_input) * self.sigmoid(q)


class SpatialConvLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, k_hop: int):
        super(SpatialConvLayer, self).__init__()
        self.g_conv = ChebNet(f_in, f_out, k_hop)
        self.activation = nn.ReLU()

        if f_in > f_out:
            self.alignement = nn.Linear(f_in, f_out)

        self.f_in, self.f_out = f_in, f_out

    def forward(self, inputs: torch.Tensor, cheb_filters: torch.Tensor) -> torch.Tensor:
        """
        Spatial graph convolution layer.
        :param inputs: tensor, [B, T, N, F_in]
        :param cheb_filters: tensor, [N, K_hop, N]
        :return: tensor, [B, T, N, F_out]
        """
        b, t, n, _ = inputs.shape

        # residual connection, first map the input to the same shape as output
        if self.f_in > self.f_out:
            ori_input = self.alignement(inputs)
        elif self.f_in < self.f_out:
            zero_shape = inputs.shape[:3] + (self.f_out - self.f_in,)
            ori_input = torch.cat([inputs, torch.zeros(zero_shape, device=inputs.device)], dim=3)
        else:
            ori_input = inputs

        outputs = self.g_conv(inputs.reshape(b * t, n, -1), cheb_filters).reshape(b, t, n, -1)
        return self.activation(outputs + ori_input)


class STConvBlock(nn.Module):
    def __init__(self,
                 k_hop: int,
                 t_cnv_krnl_sz: int,
                 channels: Tuple[int, int, int],
                 dropout: float):
        """
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param k_hop: length of Chebychev polynomial, i.e., kernel size of spatial convolution
        :param t_cnv_krnl_sz: kernel size of temporal convolution
        :param channels: three integers, define each of the sub-blocks
        :param dropout: dropout
        """
        super(STConvBlock, self).__init__()
        f_in, f_m, f_out = channels
        self.t_conv1 = TemporalConvLayer(f_in, f_m, t_cnv_krnl_sz)
        self.s_conv = SpatialConvLayer(f_m, f_m, k_hop)
        self.t_conv2 = TemporalConvLayer(f_m, f_out, t_cnv_krnl_sz)
        self.bn = nn.BatchNorm1d(f_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, cheb_filters: torch.Tensor) -> torch.Tensor:
        """
        forward of spatio-temporal convolutional block
        :param inputs: tensor, [B, T, N, F_in]
        :param cheb_filters: tensor, [N, K_hop, N]
        :return: tensor, [B, T, N, F_out]
        """
        outputs = self.t_conv2(self.s_conv(self.t_conv1(inputs), cheb_filters))
        b, t, n, _ = outputs.shape
        return self.dropout(self.bn(outputs.reshape(b * t * n, -1)).reshape(b, t, n, -1))


class StackedSTConvBlocks(nn.ModuleList):
    def forward(self, inputs: torch.Tensor, cheb_filters: torch.Tensor) -> torch.Tensor:
        h = inputs
        for module in self:
            h = module(h, cheb_filters)
        return h


class OutputLayer(nn.Module):
    def __init__(self, f_in: int, t_cnv_krnl_sz: int):
        super(OutputLayer, self).__init__()
        self.t_conv1 = TemporalConvLayer(f_in, f_in, t_cnv_krnl_sz)
        self.bn = nn.BatchNorm1d(f_in)
        self.t_conv2 = TemporalConvLayer(f_in, f_in, t_cnv_krnl_sz)
        self.out = nn.Linear(f_in, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Output layer: temporal convolution layers attach with one fully connected layer,
        which map outputs of the last st_conv block to a single-step prediction.
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [B, N, 1]
        """
        outputs = self.t_conv1(inputs)
        b, t, n, _ = outputs.shape
        return torch.mean(self.out(self.t_conv2(
            self.bn(outputs.reshape(b * t * n, -1)).reshape(b, t, n, -1))), 1)


class STGCNSingleStep(nn.Module):
    def __init__(self,
                 n_history: int,
                 k_hop: int,
                 t_cnv_krnl_sz: int,
                 dims: List[Tuple[int, int, int]],
                 dropout: float):
        super(STGCNSingleStep, self).__init__()
        self.stacked_st_blocks = StackedSTConvBlocks()
        for dim in dims:
            self.stacked_st_blocks.append(STConvBlock(k_hop, t_cnv_krnl_sz, dim, dropout))
            n_history -= 2 * (t_cnv_krnl_sz - 1)

        self.out = OutputLayer(dims[-1][-1], (n_history + 1) // 2)

    def forward(self, inputs: torch.Tensor, cheb_filters: torch.Tensor) -> torch.Tensor:
        """
        STGCN product single step prediction
        :param inputs: tensor, [B, T, N, F]
        :param cheb_filters: [N, K_hop, N]
        :return: tensor, [B, N, 1]
        """
        return self.out(self.stacked_st_blocks(inputs, cheb_filters))


class STGCN(nn.Module):
    def __init__(self, n_predictions: int, **kwargs):
        super(STGCN, self).__init__()
        self.n_predictions = n_predictions
        self.model = STGCNSingleStep(**kwargs)

    def forward(self, inputs: torch.Tensor, cheb_filters: torch.Tensor) -> torch.Tensor:
        """
        STGCN product multiple step prediction
        :param inputs: tensor, [B, T, N, F]
        :param cheb_filters: tensor, [N, K_hop, N]
        :return: tensor, [B, N_predictions, N, 1]
        """
        _, t, _, _ = inputs.shape
        h = inputs.clone()
        for i in range(self.n_predictions):
            prediction = self.model(h[:, i:i + t, ...], cheb_filters)
            h = torch.cat([h, prediction.unsqueeze(1)], 1)
        return h[:, self.n_predictions:]
