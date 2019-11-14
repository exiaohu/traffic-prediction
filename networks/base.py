import torch
from torch import nn
from torch.nn import functional as F


class ChebNet(nn.Module):
    def __init__(self, f_in: int, f_out: int, k_hop: int):
        super(ChebNet, self).__init__()
        self.theta = nn.Linear(k_hop * f_in, f_out)

    def forward(self, signals: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        """
        implement of ChebNet
        :param signals: input signals, Tensor, [B, N, F_in]
        :param supports: pre-calculated Chebychev polynomial filters, Tensor, [N, K_hop, N]
        :return: Tensor, [B, N, F_out]
        """
        b, n, _ = signals.shape
        # shape => [B, N, K, F_in]
        tmp = supports.matmul(signals.unsqueeze(1))

        # shape => [B, N, F_out]
        return self.theta(tmp.reshape(b, n, -1))


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

    def forward(self, input):
        """

        :param input: tensor, [N, C_{in}, L_{in}]
        :return: tensor, [N, C_{out}, L_{out}]
        """
        return super(CausalConv1d, self).forward(F.pad(input, [self.__padding, 0]))[:, :, :-self.__padding]
