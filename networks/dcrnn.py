import math
import random
from typing import List, Tuple

import torch
from torch import nn, Tensor

from networks.base import GraphConv


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int):
        super(DCGRUCell, self).__init__()
        self.num_node = num_node
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = GraphConv(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop)
        self.candidate_g_conv = GraphConv(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop)

    def forward(self, inputs: Tensor, supports: List[Tensor], states) -> Tuple[Tensor, Tensor]:
        r_u = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c

        return outputs, new_state


class DCRNNEncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int, n_layers: int):
        super(DCRNNEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.append(DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))

    def forward(self, inputs: Tensor, supports: List[Tensor], states: Tensor = None) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, input_size]
        :param supports: list of sparse tensors, each of shape [N, N]
        :param states: None or tensor, [n_layers, B, N, hidden_size]
        :return: tensor, [n_layers, B, N, hidden_size]
        """
        b, t, n, input_dim = inputs.shape
        if states is None:
            states = [torch.zeros(b, n, self.hidden_size, device=inputs.device, dtype=inputs.dtype)] * self.n_layers

        outputs = [None] * t
        for i_layer in range(self.n_layers):
            for i_t in range(t):
                outputs[i_t], states[i_layer] = self[i_layer](inputs[:, i_t], supports, states[i_layer])
            inputs = torch.stack(outputs, 1)
        return torch.stack(states)


class DCRNNDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, num_node: int,
                 n_supports: int, k_hop: int, n_layers: int, n_preds: int):
        super(DCRNNDecoder, self).__init__()
        self.output_size = output_size
        self.n_preds = n_preds
        self.append(DCGRUCell(output_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, supports: List[Tensor], states: Tensor,
                targets: Tensor = None, teacher_force: float = 0.5) -> Tensor:
        """
        :param supports: list of sparse tensors, each of shape [N, N]
        :param states: tensor, [n_layers, B, N, hidden_size]
        :param targets: None or tensor, [B, T, N, output_size]
        :param teacher_force: random to use targets as decoder inputs
        :return: tensor, [B, T, N, output_size]
        """
        n_layers, b, n, _ = states.shape

        inputs = torch.zeros(b, n, self.output_size, device=states.device, dtype=states.dtype)

        states = list(states)

        new_outputs = list()
        for i_t in range(self.n_preds):
            for i_layer in range(n_layers):
                inputs, states[i_layer] = self[i_layer](inputs, supports, states[i_layer])
            inputs = self.out(inputs)
            new_outputs.append(inputs)
            if targets is not None and random.random() < teacher_force:
                inputs = targets[:, i_t]

        return torch.stack(new_outputs, 1)


class DCRNN(nn.Module):
    def __init__(self,
                 n_pred: int,
                 hidden_size: int,
                 num_nodes: int,
                 n_supports: int,
                 k_hop: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 cl_decay_steps: int):
        super(DCRNN, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        self.encoder = DCRNNEncoder(input_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers)
        self.decoder = DCRNNDecoder(output_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers, n_pred)

    def forward(self, inputs: Tensor, supports: List[Tensor], targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, output_dim]
        """
        states = self.encoder(inputs, supports)
        outputs = self.decoder(supports, states, targets, self._compute_sampling_threshold(batch_seen))
        return outputs

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
