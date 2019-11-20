from typing import List

import torch
from torch import nn, Tensor, sparse
from torch.nn import functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int,
                 bias_start: float = 0.):
        super(GraphConv, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = input_dim * (max_step * n_supports + 1)
        self.weights = nn.Parameter(torch.empty(num_metrics, output_dim, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(output_dim, dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.weights, gain=1.414)
        nn.init.constant_(self.bias, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        # Reshape input and state to (batch_size, num_nodes, input_dim)
        batch_size, num_nodes, input_dim = inputs.shape

        x = inputs
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2).reshape(num_nodes, -1)  # (num_nodes, input_dim, batch_size)
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(x, shape=[-1, self._num_nodes, input_dim, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = torch.reshape(x, shape=[batch_size, self._num_nodes, -1])

        return x.matmul(self.weights) + self.bias


class ChebNet(nn.Module):
    def __init__(self, f_in: int, f_out: int, k_hop: int, bias_start: float = 0.):
        super(ChebNet, self).__init__()
        self.theta_w = nn.Parameter(torch.empty(k_hop * f_in, f_out, dtype=torch.float32), requires_grad=True)
        self.theta_b = nn.Parameter(torch.empty(f_out, dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.theta_w, gain=1.414)
        nn.init.constant_(self.theta_b, bias_start)

    def forward(self, signals: Tensor, supports: Tensor) -> Tensor:
        """
        implement of ChebNet
        :param signals: input signals, Tensor, [*, N, F_in]
        :param supports: pre-calculated Chebychev polynomial filters, Tensor, [N, K_hop, N]
        :return: Tensor, [B, N, F_out]
        """
        # shape => [B, N, K, F_in]
        tmp = supports.matmul(signals.unsqueeze(-3))
        # shape => [B, N, F_out]
        return tmp.reshape(tmp.shape[:-2] + (-1,)).matmul(self.theta_w) + self.theta_b


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, inputs):
        """
        :param inputs: tensor, [N, C_{in}, L_{in}]
        :return: tensor, [N, C_{out}, L_{out}]
        """
        outputs = super(CausalConv1d, self).forward(F.pad(inputs, [self.__padding, 0]))
        return outputs[:, :, :outputs.shape[-1] - self.__padding]
