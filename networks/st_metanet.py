from typing import List, Tuple

import numpy as np
import dgl
import torch
from dgl import DGLGraph, init
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


class MetaDense(nn.Module):
    def __init__(self, f_in: int, f_out: int, feat_size: int, meta_hiddens: List[int]):
        super(MetaDense, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.weights_mlp = MLP([feat_size] + meta_hiddens + [f_in * f_out], nn.Sigmoid, False)
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


class RNNCell(nn.Module):
    def __init__(self):
        super(RNNCell, self).__init__()

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, N, F]
        :param begin_state: None or tensor, [B, N, F]
        :return: output, tensor, [B, N, F]
                begin_state, [B, N, F]
        """
        raise NotImplementedError("Not Implemented")

    def forward(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, T, N, F]
        :param begin_state: [B, N, F]
        :return:
        """
        b, t, n, _ = data.shape

        outputs, state = list(), begin_state
        for i_t in range(t):
            output, state = self.one_step(feature, data[:, i_t], state)
            outputs.append(output)

        return torch.stack(outputs, 1), state


class MetaGRUCell(RNNCell):
    def __init__(self, f_in: int, hid_size: int, feat_size: int, meta_hiddens: List[int]):
        super(MetaGRUCell, self).__init__()
        self.hidden_size = hid_size
        self.dense_z = MetaDense(f_in + hid_size, hid_size, feat_size, meta_hiddens=meta_hiddens)
        self.dense_r = MetaDense(f_in + hid_size, hid_size, feat_size, meta_hiddens=meta_hiddens)

        self.dense_i2h = MetaDense(f_in, hid_size, feat_size, meta_hiddens=meta_hiddens)
        self.dense_h2h = MetaDense(hid_size, hid_size, feat_size, meta_hiddens=meta_hiddens)

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        b, n, _ = data.shape
        if begin_state is None:
            begin_state = torch.zeros(b, n, self.hidden_size, dtype=data.dtype, device=data.device)

        data_and_state = torch.cat([data, begin_state], -1)
        z = torch.sigmoid(self.dense_z(feature, data_and_state))
        r = torch.sigmoid(self.dense_r(feature, data_and_state))

        c = torch.tanh(self.dense_i2h(feature, data))
        h = self.dense_h2h(feature, r * begin_state)

        state = z * begin_state + torch.sub(1., z) * c + h
        return state, state


class NormalGRUCell(RNNCell):
    def __init__(self, f_in: int, hid_size: int):
        super(NormalGRUCell, self).__init__()
        self.cell = nn.GRUCell(f_in, hid_size)

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        b, n, _ = data.shape
        data = data.reshape(b * n, -1)
        if begin_state is not None:
            begin_state = begin_state.reshape(b * n, -1)
        h = self.cell(data, begin_state)
        h = h.reshape(b, n, -1)
        return h, h


class GraphAttNet(nn.Module):
    def __init__(self, dist: np.ndarray, edge: list, hid_size: int, feat_size: int,
                 meta_hiddens: List[int] = None):
        super(GraphAttNet, self).__init__()
        self.hidden_size = hid_size
        self.feature_size = feat_size
        self.meta_hiddens = meta_hiddens

        self.num_nodes = n = dist.shape[0]
        src, dst, dis = list(), list(), list()
        for i in range(n):
            for j in edge[i]:
                src.append(i)
                dst.append(j)
                dis.append(dist[i, j])

        dist = torch.tensor(dis).unsqueeze_(1)
        g = DGLGraph()
        g.set_n_initializer(init.zero_initializer)
        g.add_nodes(n)
        g.add_edges(src, dst, {'dist': dist})
        self.graph = g

    def forward(self, state: Tensor, feature: Tensor) -> Tensor:
        """
        :param state: tensor, [B, T, N, F] or [B, N, F]
        :param feature: tensor, [N, F]
        :return: tensor, [B, T, N, F]
        """
        assert state.shape[-1] == feature.shape[-1]
        # shape => [N, B, T, F] or [N, B, F]
        state = state.unsqueeze(0).transpose(0, -2).squeeze(-2)
        g = self.graph
        g.ndata['state'] = state
        g.ndata['feature'] = feature
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        return state.unsqueeze(-2).transpose(0, -2).squeeze(0)

    def msg_edge(self, edge: dgl.EdgeBatch):
        """
        :param edge: a dictionary of edge data.
            edge.src['state'] and edge.dst['state']: hidden states of the nodes, with shape [e, b, t, d] or [e, b, d]
            edge.src['feature'] and edge.dst['state']: features of the nodes, with shape [e, d]
            edge.data['dist']: distance matrix of the edges, with shape [e, d]
        :return: a dictionray of messages
        """
        raise NotImplementedError('Not implemented.')

    def msg_reduce(self, node: dgl.NodeBatch):
        """
        :param node:
                node.mailbox['state'], tensor, [n, e, b, t, d] or [n, e, b, d]
                node.mailbox['alpha'], tensor, [n, e, b, t, d] or [n, e, b, d]
        :return: tensor, [n, b, t, d] or [n, b, d]
        """
        raise NotImplementedError('Not implemented.')


class NormalGAT(GraphAttNet):
    def __init__(self, *args, **kwargs):
        super(NormalGAT, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.act = nn.LeakyReLU()

    def msg_edge(self, edge: dgl.EdgeBatch):
        state = torch.cat([edge.src['state'], edge.dst['state']], -1)

        alpha = self.act(self.linear(state))

        dist = edge.data['dist']
        while len(dist.shape) < len(alpha.shape):
            dist = torch.unsqueeze(dist, -1)

        alpha = alpha * dist
        return {'alpha': alpha, 'state': edge.src['state']}

    def msg_reduce(self, node: dgl.NodeBatch):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = torch.softmax(alpha, 1)

        new_state = torch.relu(torch.sum(alpha * state, dim=1))
        return {'new_state': new_state}


class MetaGAT(GraphAttNet):
    def __init__(self, *args, **kwargs):
        super(MetaGAT, self).__init__(*args, **kwargs)
        self.w_mlp = MLP([self.hidden_size * 2 + 1] + self.meta_hiddens + [self.hidden_size * 2 * self.hidden_size],
                         nn.Sigmoid, False)
        self.act = nn.LeakyReLU()
        self.weight = torch.zeros(1, 1)

    def msg_edge(self, edge: dgl.EdgeBatch):
        state = torch.cat([edge.src['state'], edge.dst['state']], -1)  # [X, B, T, 2F] or [X, B, 2F]
        feature = torch.cat([edge.src['feature'], edge.dst['feature'], edge.data['dist']], -1)  # [X, 2F + 1]

        weight = self.w_mlp(feature).reshape(-1, self.hidden_size * 2, self.hidden_size)  # [X, 2H, H]

        shape = state.shape
        state = state.reshape(shape[0], -1, shape[-1])

        # [X, ?, 2F] * [X. 2H, H] => [X, ?, H]
        alpha = self.act(torch.bmm(state, weight))

        alpha = alpha.reshape(*shape[:-1], self.hidden_size)
        return {'alpha': alpha, 'state': edge.src['state']}

    def msg_reduce(self, node: dgl.NodeBatch):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = torch.softmax(alpha, 1)

        new_state = torch.relu(torch.sum(alpha * state, dim=1)) * torch.sigmoid(self.weight)
        return {'new_state': new_state}


class STMetaEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size1: int, hidden_size2: int, meta_hiddens: List[int],
                 graph: Tuple[np.ndarray, list, list], gat_hidden_size: int):
        super(STMetaEncoder, self).__init__()

        self.gru1 = NormalGRUCell(input_dim, hidden_size1)
        dist, e_in, e_out = graph
        self.g1 = MetaGAT(dist.T, e_in, gat_hidden_size, meta_hiddens)
        self.g2 = MetaGAT(dist, e_out, gat_hidden_size, meta_hiddens)
        self.gru2 = MetaGRUCell(hidden_size1, hidden_size2, feat_size, meta_hiddens)

    def forward(self, feature: Tensor, data: Tensor) -> List[Tensor]:
        data, state1 = self.gru1(feature, data)
        print(data.shape, state1.shape)
        data = self.g2(data, feature) + self.g1(data, feature)
        print(data.shape)
        data, state2 = self.gru2(feature, data)
        print(data.shape, state2.shape)
        return [state1, state2]


class STMetaDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size1: int, hidden_size2: int, meta_hiddens: List[int],
                 graph: Tuple[np.ndarray, list, list], gat_hidden_size: int):
        super(STMetaDecoder, self).__init__()
        pass

    def forward(self, feature: Tensor, data: Tensor) -> List[Tensor]:
        pass

# def test():
#     dist = np.random.randn(207, 207)
#     edge1, edge2 = [[] for _ in range(207)], [[] for _ in range(207)]
#     for i in range(207):
#         for j in range(207):
#             if np.random.random() < 0.2:
#                 edge1[i].append(j)
#                 edge2[j].append(i)
#     m = STMetaEncoder(2, 32, 32, [32, 4], (dist, edge1, edge2), 32)
#     data = torch.randn(31, 12, 207, 2)
#     feature = torch.randn(207, 32)
#     states = m(feature, data)
#     print(states[0].shape, states[1].shape)
