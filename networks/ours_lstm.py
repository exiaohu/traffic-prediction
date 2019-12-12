from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, support_len: int, order: int):
        super(GraphConv, self).__init__()
        self.order = order
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x: Tensor, supports: Tensor):
        """
        :param x: tensor, [B, *, N, c_in]
        :param supports: tensor, [n_edge, N, N] or [n_edge, B, N, N]
        :return: tensor, [B, *, N, c_out]
        """
        out = [x]
        for support in supports:
            x1 = self.nconv(x, support)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, support)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, -1)
        h = self.mlp(h)
        return h

    @staticmethod
    def nconv(x: Tensor, adj: Tensor):
        assert len(adj.shape) in [2, 3] and len(x.shape) in [3, 4], f'x of {x.shape} or adj of {adj.shape} is wrong.'
        x_, r_ = ('btvc', 'btwc') if len(x.shape) == 4 else ('bvc', 'bwc')
        a_ = 'vw' if len(adj.shape) == 2 else 'bvw'
        x = torch.einsum(f'{x_},{a_}->{r_}', [x, adj])
        return x.contiguous()


class StackedGraphConv(nn.ModuleList):
    def __init__(self, n_layers: int, input_dim: int, hidden_size: int, edge_dim: int, order: int):
        super(StackedGraphConv, self).__init__()
        self.append(GraphConv(input_dim, hidden_size, edge_dim, order))
        for _ in range(1, n_layers):
            self.append(GraphConv(hidden_size, hidden_size, edge_dim, order))

    def forward(self, x: Tensor, supports: Tensor):
        for gc in self:
            x = gc(x, supports)
            x = F.relu(x)
        return x


class OursLSTM(nn.Module):
    def __init__(self,
                 n_hist: int,
                 n_pred: int,
                 n_graphconv: int,
                 hidden_size: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 num_node: int,
                 edge_dim: int,
                 node_dim: int,
                 order: int,
                 dropout: float):
        super(OursLSTM, self).__init__()
        self.dropout = dropout
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.num_node = num_node
        self.output_dim = output_dim

        self.vertexes = nn.Parameter(torch.randn(num_node, node_dim), requires_grad=True)
        self.arcs = nn.Sequential(
            nn.Linear(2 * node_dim, 4 * node_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * node_dim, 4 * node_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * node_dim, edge_dim),
        )

        self.encoder_pre = StackedGraphConv(n_graphconv, input_dim, hidden_size, edge_dim, order)
        self.encoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)

        self.decoder_pre = StackedGraphConv(n_graphconv, output_dim, hidden_size, edge_dim, order)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_rnn_layers)
        self.decoder_post = StackedGraphConv(n_graphconv, hidden_size, hidden_size, edge_dim, order)
        self.projector = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        """
        dynamic convoluitonal recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, [B, n_pred, N, input_dim]
        """
        supports = self.adaptive_supports(inputs.device)

        h, c = self.encoding(inputs, supports)

        torch.dropout(h, p=self.dropout, train=self.training)
        torch.dropout(c, p=self.dropout, train=self.training)

        outputs = self.decoding((h, c), supports, targets)
        return outputs

    def encoding(self, inputs: Tensor, supports: Tensor) -> Tuple[Tensor, Tensor]:
        """
        encoding
        :param inputs: tensor, [B, n_hist, N, input_dim]
        :param supports: tensor, [edge_dim, N, N]
        :return: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        """
        b, _, n, input_dim = inputs.shape

        inputs = self.encoder_pre(inputs, supports)
        inputs = inputs.transpose(0, 1).reshape(self.n_hist, b * n, -1)
        _, (h, c) = self.encoder(inputs)
        return h, c

    def decoding(self, hc: Tuple[Tensor, Tensor], supports: Tensor, targets: Optional[Tensor]):
        """
        decoding
        :param hc: 2-tuple tensor, each with shape [n_rnn_layers, B * N, hidden_size]
        :param supports: tensor, [edge_dim, N, N]
        :param targets: optional, exists while training, tensor, [B, n_pred, N, output_dim]
        :return: tensor, shape as same as targets
        """
        h, c = hc
        b, n = h.shape[1] // self.num_node, self.num_node
        decoder_input = torch.zeros(b, n, self.output_dim, device=h.device, dtype=h.dtype)

        outputs = list()
        for t in range(self.n_pred):
            decoder_input = self.decoder_pre(decoder_input, supports)
            decoder_input = torch.reshape(decoder_input, [1, b * n, -1])
            decoder_input, (h, c) = self.decoder(decoder_input, (h, c))
            decoder_input = torch.reshape(decoder_input, [b, n, -1])
            decoder_input = self.decoder_post(decoder_input, supports)
            decoder_input = self.projector(decoder_input)
            outputs.append(decoder_input)
            if targets is not None:
                decoder_input = targets[:, t]
        return torch.cat(outputs).reshape(self.n_pred, b, n, -1).transpose(0, 1)

    def adaptive_supports(self, device: torch.device) -> Tensor:
        num_node, node_dim = self.vertexes.shape
        vertexes = self.vertexes
        src = vertexes.unsqueeze(0).expand([num_node, num_node, node_dim])
        dst = vertexes.unsqueeze(1).expand([num_node, num_node, node_dim])
        adj_mxs = self.arcs(torch.cat([src, dst], -1)).permute([2, 0, 1])

        identity = torch.eye(num_node, dtype=torch.float32, device=device)
        adj_mxs = F.normalize(F.relu(adj_mxs.contiguous()), p=1, dim=2)
        # adaptive_supports = adj_mxs + identity
        adaptive_supports = identity.unsqueeze(0).expand(adj_mxs.shape)

        return adaptive_supports


def test():
    # from networks.ours_lstm import OursLSTM
    m = OursLSTM(12, 12, 3, 32, 2, 2, 1, 207, 4, 4, 2, 0.0)
    import torch
    x, y = torch.randn(64, 12, 207, 2), torch.randn(64, 12, 207, 1)
    y_ = m(x, y)
    print(y_.shape)
