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
        self.weights_mlp = MLP(meta_hiddens + (f_in * f_out,), nn.Sigmoid, False)
        self.bias_mlp = MLP(meta_hiddens + (f_out,), nn.Sigmoid, False)

    def forward(self, meta_knowledge: Tensor, inputs: Tensor) -> Tensor:
        weights = self.weights_mlp(meta_knowledge).reshape(-1, self.f_in, self.f_out)
        bias = self.bias_mlp(meta_knowledge)

        return inputs.bmm(weights) + bias
