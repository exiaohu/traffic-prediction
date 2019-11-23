from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# class GraphConvNet(nn.Module):
#     def __init__(self, f_in: int, f_out: int, n_matrices: int):
#         super(GraphConvNet, self).__init__()
#         self.f_in, self.f_out = f_in, f_out
#         self.chan_wise = nn.Linear(f_in, f_out)
#         self.mtrx_wise = nn.Linear(n_matrices, 1)
#
#     def forward(self, signals: Tensor, supports: Tensor) -> Tensor:
#         """
#         :param signals: input signals, Tensor, [B, *, N, F_in]
#         :param supports: Tensor, [B, N, n_matrices, N]
#         :return: Tensor, [B, N, F_out]
#         """
#         signals = signals.unsqueeze(-3)
#         while len(supports.shape) < len(signals.shape):
#             supports = supports.unsqueeze(1)
#         if self.f_in > self.f_out:
#             # shape => [B, N, F_out]
#             signals = self.chan_wise(signals)
#             # shape => [B, N, n_matrices, F_out]
#             outputs = supports.matmul(signals)
#         else:
#             # shape => [B, N, n_matrices, F_in]
#             outputs = supports.matmul(signals)
#             # shape => [B, N, n_matrices, F_out]
#             outputs = self.chan_wise(outputs)
#         # shape => [B, N, n_matrices, F_out]
#         outputs = self.mtrx_wise(outputs.transpose(-1, -2)).squeeze(-1)
#         return outputs
class GraphConvNet(nn.Module):
    def __init__(self, f_in: int, f_out: int, n_matrices: int):
        super(GraphConvNet, self).__init__()
        self.out = nn.Linear(n_matrices * f_in, f_out)

    def forward(self, signals: Tensor, supports: Tensor) -> Tensor:
        """
        implement of ChebNet
        :param signals: input signals, Tensor, [B, *, N, F_in]
        :param supports: tensor, [B, N, n_matrices, N]
        :return: Tensor, [B, *, N, F_out]
        """
        b, n, n_matrices, _ = supports.shape
        supports = supports.reshape(b, -1, n)
        while len(supports.shape) < len(signals.shape):
            supports.unsqueeze_(1)

        out = torch.matmul(supports, signals)  # [B, *, N*n_matrices, F_in]
        out = out.reshape(signals.shape[:-2] + (n, -1))  # [B, *, N, n_matrices*F_in]
        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(self, n_hidden: int):
        super(SelfAttention, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, inputs: Tensor):
        """
        :param inputs: tensor, [B, n_hist, N, n_hidden]
        :return: tensor, [B, N, n_hidden]
        """
        energy = self.projector(inputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        outputs = (inputs * weights.unsqueeze(-1)).sum(1)
        return outputs


class Ours(nn.Module):
    def __init__(self,
                 num_node: int,
                 n_in: int,
                 n_hidden: int,
                 n_out: int,
                 node_dim: int,
                 edge_dim: int,
                 n_pred: int):
        super(Ours, self).__init__()
        self.n_pred, self.n_out = n_pred, n_out
        self.ne = nn.Embedding(num_node, node_dim)
        self.n2e = nn.Linear(node_dim, edge_dim * node_dim)

        self.gcn1 = GraphConvNet(n_in, n_hidden, edge_dim)
        self.gcn2 = GraphConvNet(n_hidden, n_hidden, edge_dim)

        self.bn = nn.BatchNorm2d(n_hidden)
        self.bn1 = nn.BatchNorm2d(n_hidden)

        self.bn2 = nn.BatchNorm1d(n_hidden)

        self.sa = SelfAttention(n_hidden)

        self.encoder = nn.GRU(n_hidden, n_hidden)

        self.gru = nn.GRUCell(n_out, n_hidden)

        self.out1 = GraphConvNet(n_hidden, n_hidden, edge_dim)
        self.out2 = GraphConvNet(n_hidden, n_out, edge_dim)

        self.bn3 = nn.BatchNorm1d(n_hidden)
        self.bn4 = nn.BatchNorm1d(n_out)

        self.s_ne = nn.Sequential(
            nn.Linear(n_in, node_dim),
            nn.ReLU(True),
            SelfAttention(node_dim)
        )

    def forward(self, inputs: Tensor, targets: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        the forward of our model
        :param inputs: tensor, [B, n_hist, N, input_dim]
        :param targets: tensor, [B, n_pred, N, output_dim]
        :return: outputs, tensor, [B, n_pred, N, output_dim]
                supports, tensor, [B, N, edge_dim, N]
        """
        b, n_hist, n, _ = inputs.shape

        supports = self.compute_supports(inputs)

        inputs = self.gcn1(inputs, supports)  # [B, n_hist, N, hidden_size]
        inputs = torch.relu(inputs)
        inputs = self.bn(torch.transpose(inputs, 1, 3)).transpose(1, 3)
        inputs = self.gcn2(inputs, supports)  # [B, n_hist, N, hidden_size]
        inputs = torch.relu(inputs)
        inputs = self.bn(torch.transpose(inputs, 1, 3)).transpose(1, 3)
        inputs, _ = self.encoder(inputs.transpose(0, 1).reshape(n_hist, b * n, -1))
        inputs = inputs.reshape(n_hist, b, n, -1).transpose(0, 1)

        return self.decoding(inputs, supports, targets), supports

    def compute_supports(self, inputs: Tensor) -> Tensor:
        """
        :param inputs: tensor, [B, n_hist, N, input_dim]
        :return: tensor, [B, N, edge_dim, N]
        """
        b, _, n, _ = inputs.shape
        device = inputs.device

        ne = self.ne(torch.tensor([range(n)], device=device))  # [N, node_dim]

        ne = ne.repeat(b)

        # ne_s = self.s_ne(inputs)  # [B, N, node_dim]
        #
        # ne = (ne + ne_s).unsqueeze(1)  # [B, 1, N, node_dim]

        ee = self.n2e(ne).reshape(b, n, -1, self.ne.embedding_dim)  # [B, N, edge_dim, node_dim]

        ls = torch.matmul(ee, ne.transpose(-2, -1))  # [B, N, edge_dim, N]
        ls = F.normalize(torch.abs(ls), p=1, dim=-1)  # [B, N, edge_dim, N]

        e = torch.eye(n, device=device, dtype=inputs.dtype)
        return (ls.transpose(1, 2) + e).transpose(1, 2)  # [B, N, edge_dim, N]

    def decoding(self, historical_states: Tensor, supports: Tensor, targets: Tensor = None) -> Tensor:
        """
        decoding
        :param historical_states: tensor, [B, n_hist, N, hidden_size]
        :param supports: tensor, [B, N, edge_dim, N]
        :param targets: optional, exists in training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, shape as same as targets
        """
        b, _, n, _ = historical_states.shape

        go_symbol = torch.zeros((b * n, self.n_out), device=historical_states.device, dtype=historical_states.dtype)

        decoder_inputs = go_symbol

        outputs = list()
        for t in range(self.n_pred):
            hidden_state = self.sa(historical_states)
            output = self.gru(decoder_inputs.reshape(b * n, -1), hidden_state.reshape(b * n, -1)).reshape(b, n, -1)
            historical_states = torch.cat([historical_states, output.unsqueeze(1)], 1)
            output = self.bn2(torch.relu(output).transpose(1, 2)).transpose(1, 2)

            decoder_inputs = self.bn3(torch.relu(self.out1(output, supports)).transpose(1, 2)).transpose(1, 2)
            decoder_inputs = self.bn4(torch.relu(self.out2(decoder_inputs, supports)).transpose(1, 2)).transpose(1, 2)
            outputs.append(decoder_inputs)
            if targets is not None:
                decoder_inputs = targets[:, t, ...]
        return torch.stack(outputs, 1)
