from typing import Tuple, List

import numpy as np
import torch
from torch import nn, Tensor

from .base import GraphConv


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int,
                 proj_num: int = None):
        super(DCGRUCell, self).__init__()
        self.num_node = num_node
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = GraphConv(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop,
                                        bias_start=1.0)
        self.candidate_g_conv = GraphConv(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop)

        if proj_num is not None:
            self.out = nn.Linear(hidden_size, proj_num)

    def forward(self, inputs: Tensor, supports: List[Tensor], states: Tensor = None) -> Tuple[Tensor, Tensor]:
        ru = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = ru.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c

        if hasattr(self, 'out'):
            outputs = self.out(outputs)

        return outputs, new_state


class StackedDCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int,
                 n_rnn_layers: int, proj_num: int = None):
        super(StackedDCGRUCell, self).__init__()
        self.n_rnn_layers = n_rnn_layers
        self.hidden_size = hidden_size
        dcgrus = [DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop)]
        for i, _ in enumerate(range(1, n_rnn_layers - 1)):
            dcgrus.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))
        dcgrus.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop, proj_num=proj_num))

        self.dcgrus = nn.ModuleList(dcgrus)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor],
                hidden_states: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Encoder forward pass.
        :param inputs: tensor, [B, N, input_size]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param hidden_states: tensor, [n_rnn_layers, B, N, hidden_size]
        :return: output: [B, N, hidden_size)
                 hidden_state [n_rnn_layers, B, N, hidden_size]
                 (lower indices mean lower layers)
        """
        if hidden_states is None:
            zero_size = (self.n_rnn_layers,) + inputs.shape[:2] + (self.hidden_size,)
            hidden_states = torch.zeros(zero_size, dtype=inputs.dtype, device=inputs.device)
        outputs, new_hidden_states = inputs, list()
        for idx, dcgru in enumerate(self.dcgrus):
            outputs, hidden_state = dcgru(outputs, supports, hidden_states[idx])
            new_hidden_states.append(hidden_state)

        return outputs, torch.stack(new_hidden_states)


class DCRNN(nn.Module):
    def __init__(self,
                 n_hist: int,
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
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.output_dim = output_dim
        self.encoder = StackedDCGRUCell(input_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers)
        self.decoder = StackedDCGRUCell(output_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers,
                                        proj_num=output_dim)

    def forward(self, inputs: Tensor, supports: List[Tensor], targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, input_dim]
        """
        encoder_hidden_states = self.encoding(inputs.transpose(0, 1), supports)
        outputs = self.decoding(encoder_hidden_states,
                                supports,
                                targets.transpose(0, 1) if targets is not None else None,
                                batch_seen)
        return outputs.transpose(0, 1)

    def encoding(self, inputs: Tensor, supports: List[Tensor]) -> Tensor:
        """
        encoding
        :param inputs: tensor, [n_hist, B, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :return: tensor, [n_rnn_layers, B, N, hidden_size]
        """
        encoder_hidden_states = None
        for t in range(self.n_hist):
            _, encoder_hidden_states = self.encoder(inputs[t], supports, encoder_hidden_states)

        return encoder_hidden_states

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def decoding(self,
                 encoder_hidden_states: Tensor,
                 supports: List[Tensor],
                 targets: Tensor = None,
                 batch_seen: int = None):
        """
        decoding
        :param encoder_hidden_states: tensor, [n_rnn_layers, B, N, hidden_size]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: optional, exists while training, tensor, [n_pred, B, N, output_dim]
        :return: tensor, shape as same as targets
        :param batch_seen: the number of batches model seen
        """
        _, b, n, _ = encoder_hidden_states.shape
        go_symbol = torch.zeros((b, n, self.output_dim),
                                device=encoder_hidden_states.device,
                                dtype=encoder_hidden_states.dtype)

        decoder_hidden_states = encoder_hidden_states
        decoder_input = go_symbol

        outputs = list()
        for t in range(self.n_pred):
            decoder_input, decoder_hidden_states = self.decoder(decoder_input,
                                                                supports,
                                                                decoder_hidden_states)
            outputs.append(decoder_input)
            if targets is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batch_seen):
                    decoder_input = targets[t]
        return torch.stack(outputs)
