from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor

from .base import ChebNet


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, k_hop: int, proj_num: int = None):
        super(DCGRUCell, self).__init__()
        self.num_node = num_node
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = ChebNet(input_size + hidden_size, hidden_size * 2, k_hop)
        self.candidate_g_conv = ChebNet(input_size + hidden_size, hidden_size, k_hop)

        if proj_num is not None:
            self.out = nn.Linear(hidden_size, proj_num)

    def forward(self, inputs: Tensor, graph: Tensor, states: Tensor = None) -> Tuple[Tensor, Tensor]:
        if states is None:
            zero_size = inputs.shape[:2] + (self.hidden_size,)
            states = torch.zeros(zero_size, dtype=inputs.dtype, device=inputs.device)

        ru = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), graph))
        r, u = ru.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), graph))
        new_state = u * states + (1 - u) * c

        outputs = new_state
        if hasattr(self, 'out'):
            outputs = self.out(outputs)

        return outputs, new_state


class StackedDCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, k_hop: int, n_rnn_layers: int,
                 proj_num: int = None):
        super(StackedDCGRUCell, self).__init__()
        self.n_rnn_layers = n_rnn_layers
        self.hidden_size = hidden_size
        dcgrus = [DCGRUCell(input_size, hidden_size, num_node, k_hop)]
        for _ in range(1, n_rnn_layers - 1):
            dcgrus.append(DCGRUCell(hidden_size, hidden_size, num_node, k_hop))
        if proj_num is not None:
            dcgrus.append(DCGRUCell(hidden_size, hidden_size, num_node, k_hop, proj_num=proj_num))
        else:
            dcgrus.append(DCGRUCell(hidden_size, hidden_size, num_node, k_hop))

        self.dcgrus = nn.ModuleList(dcgrus)

    def forward(self,
                inputs: Tensor,
                graph_filters: Tensor,
                hidden_states: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Encoder forward pass.
        :param inputs: tensor, [B, N, input_size]
        :param graph_filters: tensor, [N, K_hop, N]
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
            outputs, hidden_state = dcgru(outputs, graph_filters, hidden_states[idx])
            new_hidden_states.append(hidden_state)

        return outputs, torch.stack(new_hidden_states)


class DCRNN(nn.Module):
    def __init__(self,
                 n_hist: int,
                 n_pred: int,
                 hidden_size: int,
                 num_nodes: int,
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
        self.encoder = StackedDCGRUCell(input_dim, hidden_size, num_nodes, k_hop, n_rnn_layers)
        self.decoder = StackedDCGRUCell(output_dim, hidden_size, num_nodes, k_hop, n_rnn_layers, proj_num=output_dim)

    def forward(self, inputs: Tensor, graph_filters: Tensor, targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param graph_filters: [N, K_hop, N]
        :param targets: exists for training, tensor, [B, n_hist, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, input_dim]
        """
        encoder_hidden_states = self.encoding(inputs.transpose(0, 1), graph_filters)
        outputs = self.decoding(encoder_hidden_states,
                                graph_filters,
                                targets.transpose(0, 1) if targets is not None else None,
                                batch_seen)
        return outputs.transpose(0, 1)

    def encoding(self, inputs: Tensor, graph_filters: Tensor) -> Tensor:
        """
        encoding
        :param inputs: tensor, [n_hist, B, N, input_dim]
        :param graph_filters: tensor, [N, K_hop, N]
        :return: tensor, [n_rnn_layers, B, N, hidden_size]
        """
        encoder_hidden_states = None
        for t in range(self.n_hist):
            _, encoder_hidden_states = self.encoder(inputs[t], graph_filters, encoder_hidden_states)

        return encoder_hidden_states

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def decoding(self,
                 encoder_hidden_states: Tensor,
                 graph_filters: Tensor,
                 targets: Tensor = None,
                 batch_seen: int = None):
        """
        decoding
        :param encoder_hidden_states: tensor, [n_rnn_layers, B, N, hidden_size]
        :param graph_filters: [N, K_hop, N]
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
                                                                graph_filters,
                                                                decoder_hidden_states)
            outputs.append(decoder_input)
            if targets is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batch_seen):
                    decoder_input = targets[t]
        return torch.stack(outputs)
