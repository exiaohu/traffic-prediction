from typing import List

from torch import nn, Tensor


class MLP(nn.Sequential):
    def __init__(self, hiddens: List[int], hidden_act, out_act: bool):
        super(MLP, self).__init__()
        for i in range(len(hiddens)):
            if i == 0:
                continue
            self.add_module(f'Layer{i}', nn.Linear(hiddens[i - 1], hiddens[i]))
            if i < len(hiddens) - 1 or out_act:
                self.add_module(f'Activation{i}', hidden_act())


class MetaLinear(nn.Module):
    def __init__(self, f_in: int, f_out: int, meta_hiddens: List[int]):
        super(MetaLinear, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.weights_mlp = MLP(meta_hiddens + [f_in * f_out], nn.Sigmoid, False)
        self.bias_mlp = MLP(meta_hiddens + [f_out], nn.Sigmoid, False)

    def forward(self, feature: Tensor, data: Tensor) -> Tensor:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, N, F_in]
        :return:
        """
        b, n, _ = data.shape
        data = data.reshape(b, n, 1, self.f_in)
        weights = self.weights_mlp(feature).reshape(1, n, self.f_in, self.f_out)  # [F_in, F_out]
        bias = self.bias_mlp(feature)  # [n, F_out]

        return data.matmul(weights).squeeze(2) + bias
