import math
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import dgl


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, node_dim: int, edge_dim: int, order: int, dropout: float):
        super(GraphConv, self).__init__()
        self.order = order
        c_in = (order * edge_dim + 1) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = nn.Dropout(dropout)

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

        h = torch.cat(out, 1)
        h = self.mlp(h)
        h = self.dropout(h)
        return h

    @staticmethod
    def nconv(x: Tensor, adj: Tensor):
        assert len(adj.shape) in [2, 3] and len(x.shape) in [3, 4], f'x of {x.shape} or adj of {adj.shape} is wrong.'
        x_, r_ = ('bfvc', 'bfwc') if len(x.shape) == 4 else ('bvc', 'bwc')
        a_ = 'vw' if len(adj.shape) == 2 else 'bvw'
        x = torch.einsum(f'{x_},{a_}->{r_}', [x, adj])
        return x.contiguous()


# class GraphConv(nn.Module):
#     def __init__(self, c_in: int, c_out: int, node_dim: int, edge_dim: int, order: int, dropout: float):
#         super(GraphConv, self).__init__()
#         self.order = order
#         self.c_in, self.c_out = c_in, c_out
#         # c_in = (order * edge_dim + 1) * c_in
#         # self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
#         self.w_mlp = nn.Linear(2 * node_dim, c_in * c_out)
#         self.b_mlp = nn.Linear(2 * node_dim, c_out)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x: Tensor, supports: dgl.DGLGraph) -> Tensor:
#         supports = supports.local_var()
#         supports.ndata['x'] = x.transpose(0, 2)
#         supports.update_all(message_func=self.message_func,
#                             reduce_func=self.reduce_func)
#         return supports.ndata.pop('h').transpose(0, 2)
#
#     def message_func(self, edges: dgl.EdgeBatch):
#         feats = torch.cat([edges.src['feats'], edges.dst['feats']], -1)  # [E, 2F]
#         ne, dim = feats.shape
#         x = edges.src['x']  # [E, C, B, T]
#         w = self.w_mlp(feats).reshape(ne, self.c_in, self.c_out)
#         b = self.b_mlp(feats)
#         msg = torch.einsum('ecbt,ecf,ef->efbt', [x, w, b])
#         sim = edges.data.get('sim').reshape(-1, 1, 1, 1)  # [E, 1]
#         return {'msg': msg, 'sim': sim}
#
#     def reduce_func(self, nodes: dgl.NodeBatch):
#         # sim = nodes.mailbox['sim']  # [n, e, 1, 1, 1]
#         msg = nodes.mailbox['msg']  # [n, e, f, b, t]
#         # sim = F.softmax(sim, 1)
#         return {'h': torch.sum(msg, 1)}


class StackedGraphConv(nn.ModuleList):
    def __init__(self, dims: List[int], node_dim: int, edge_dim: int, order: int, dropout: float):
        super(StackedGraphConv, self).__init__()
        for i, dim in enumerate(dims):
            if i == 0:
                continue
            self.append(GraphConv(dims[i - 1], dim, node_dim, edge_dim, order, dropout))

    def forward(self, inputs: Tensor, supports: Tensor):
        """
        : param inputs: tensor, [B, F, N, T]
        : param supports: tensor, [n_edge, N, N] or [n_edge, B, N, N]
        """
        x = inputs
        for gc in self:
            x = gc(x, supports)
            x = torch.relu(x)
        return x


class STLayer(nn.Module):
    def __init__(self, n_residuals: int, n_dilations: int, kernel_size: int, dilation: int,
                 n_skip: int, node_dim: int, edge_dim: int, order: int, dropout: float):
        super(STLayer, self).__init__()
        # dilated convolutions
        self.filter_conv = nn.Conv2d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)

        self.gate_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)

        # 1x1 convolution for residual connection
        self.gconv = GraphConv(n_dilations, n_residuals, node_dim, edge_dim, order, dropout)

        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(n_dilations, n_skip, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(n_residuals)

    def forward(self, x: Tensor, skip: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        residual = x
        # dilated convolution
        _filter = self.filter_conv(residual)
        _filter = torch.tanh(_filter)
        _gate = self.gate_conv(residual)
        _gate = torch.sigmoid(_gate)
        x = _filter * _gate

        # parametrized skip connection
        s = x
        s = self.skip_conv(s)
        skip = skip[:, :, :, -s.size(3):]
        skip = s + skip

        x = self.gconv(x, supports)

        x = x + residual[:, :, :, -x.size(3):]

        x = self.bn(x)
        return x, skip


class STBlock(nn.ModuleList):
    def __init__(self, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, node_dim: int, edge_dim: int, order: int, dropout: float):
        super(STBlock, self).__init__()
        for i in range(n_layers):
            self.append(
                STLayer(n_residuals, n_dilations, kernel_size, 2 ** i, n_skips, node_dim, edge_dim, order, dropout))

    def forward(self, x: Tensor, skip: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        for layer in self:
            x, skip = layer(x, skip, supports)

        return x, skip


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, n_blocks, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, node_dim: int, edge_dim: int, order: int, dropout: float):
        self.n_skips = n_skips
        super(StackedSTBlocks, self).__init__()
        for _ in range(n_blocks):
            self.append(
                STBlock(n_layers, kernel_size, n_residuals, n_dilations, n_skips, node_dim, edge_dim, order, dropout))

    def forward(self, x: Tensor, supports: Union[Tensor, dgl.DGLGraph]):
        b, f, n, t = x.shape
        skip = torch.zeros(b, self.n_skips, n, t, dtype=torch.float32, device=x.device)
        for block in self:
            x, skip = block(x, skip, supports)
        return x, skip


class Ours(nn.Module):
    def __init__(self,
                 device,
                 factor: np.ndarray,
                 dynamic_bias: Optional[float],
                 num_node: int,
                 n_in: int,
                 n_out: int,
                 n_hist: int,
                 n_pred: int,
                 node_dim: int,
                 edge_dim: int,
                 n_residuals: int,
                 n_dilations: int,
                 n_skips: int,
                 n_ends: int,
                 kernel_size: int,
                 n_blocks: int,
                 n_layers: int,
                 order: int,
                 dropout: float):
        super(Ours, self).__init__()
        self.factor = torch.tensor(factor, dtype=torch.float32, device=device)
        self.factor2feats = nn.Linear(factor.shape[-1], node_dim)
        self.dynamic_bias = dynamic_bias

        self.receptive_field = n_blocks * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        self.enter = nn.Conv2d(n_in * 2, n_residuals, kernel_size=(1, 1))

        self.blocks = StackedSTBlocks(n_blocks, n_layers, kernel_size, n_residuals, n_dilations,
                                      n_skips, node_dim, edge_dim, order, dropout)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_skips, n_ends, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_ends, n_pred, kernel_size=(1, 1))
        )

        # self.vertexes = nn.Parameter(torch.randn(num_node, node_dim), requires_grad=True)
        # self.adaptive = nn.ParameterDict()
        # self.adaptive['weight'] = nn.Parameter(torch.ones(node_dim, dtype=torch.float32))
        # self.adaptive['lambda'] = nn.Parameter(torch.randn(node_dim, node_dim, edge_dim, dtype=torch.float32))
        self.adaptive = nn.ModuleDict({
            'mapping': nn.Sequential(
                nn.Linear(self.factor.shape[-1], node_dim),
                nn.ReLU()
            ),
            'arcs': nn.Sequential(
                nn.Linear(2 * node_dim, 2 * node_dim),
                nn.ReLU(),
                nn.Linear(2 * node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, edge_dim)
            )
        })

        if dynamic_bias is not None:
            self.dynamic = nn.ModuleDict({
                'time-reduce': nn.Conv1d(n_hist, 1, kernel_size=(1, 1)),
                'arcs': nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(4 * n_in, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, edge_dim)
                )
            })

    def forward(self, inputs: Tensor):
        """
        : params inputs: tensor, [B, T, N, F]
        """
        supports = self.adaptive_supports()
        # supports = (self.sim_matrix(self.factor).unsqueeze(0) + 1.0) / 2.0
        # supports = self.adaptive_supports(self.vertexes)
        # supports = self.get_graph()

        static_x, dynamic_x = self.inputs_split(inputs)

        inputs = torch.cat([static_x, dynamic_x], -1)

        # if self.dynamic_bias is not None:
        # dynamic_bias = self.compute_dynamic_bias(inputs)
        # supports = torch.relu(supports.unsqueeze(1) + self.dynamic_bias * dynamic_bias)

        inputs = inputs.transpose(1, 3)  # [B, F, N, T]
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = F.pad(inputs, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = inputs
        x = self.enter(x)

        x, skip = self.blocks(x, supports)

        x = self.out(skip)
        return x, supports

    def get_graph(self, threshold: float = 0.4) -> dgl.DGLGraph:
        feats = self.factor
        n_nodes, _ = self.factor.shape
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        sims = self.sim_matrix(feats)
        edges = (sims.abs() >= threshold).nonzero()
        sims = sims[edges[:, 0], edges[:, 1]].unsqueeze(-1)
        g.add_edges(edges[:, 0], edges[:, 1], {'sim': sims})
        g.add_edges(edges[:, 1], edges[:, 0], {'sim': sims})

        g.ndata['feats'] = self.factor2feats(feats)
        return g

    @staticmethod
    def sim_matrix(x: Tensor, eps=1e-8) -> Tensor:
        x_n = x.norm(dim=1)[:, None]
        x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n, device=x.device))
        sim_mt = torch.mm(x_norm, x_norm.transpose(0, 1))
        return sim_mt

    # @property
    # def adaptive_supports(self) -> Tensor:
    #     return torch.eye(self.graph.number_of_nodes(), device=self.factor.device, dtype=torch.float32)

    def adaptive_supports(self) -> Tensor:
        vertexes = self.adaptive['mapping'](self.factor)
        num_node, node_dim = vertexes.shape
        src = vertexes.unsqueeze(0).expand([num_node, num_node, node_dim])
        dst = vertexes.unsqueeze(1).expand([num_node, num_node, node_dim])
        adj_mxs = self.adaptive['arcs'](torch.cat([src, dst], -1)).permute([2, 0, 1])

        # vertexes = vertexes * self.adaptive['weight']
        # adj_mxs = torch.einsum('nd,dqe,qv->env', [vertexes, self.adaptive['lambda'], vertexes.t()])

        identity = torch.eye(num_node, dtype=torch.float32, device=vertexes.device)
        # adj_mxs = F.normalize(F.relu(adj_mxs.contiguous()), p=1, dim=2)
        adj_mxs = torch.softmax(adj_mxs.contiguous(), dim=2)
        adaptive_supports = torch.max(adj_mxs, identity)
        # adaptive_supports[adaptive_supports < 0.1] = 0.0
        # adaptive_supports = identity.unsqueeze(0).expand(adj_mxs.shape)

        return adaptive_supports

    def inputs_split(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        static_x = torch.einsum('btnf,nu,up->btpf', [inputs, self.factor, self.factor.t()])
        return static_x, inputs - static_x

    def compute_dynamic_bias(self, dynamics: Tensor) -> Tensor:
        b, t, n, f = dynamics.shape
        dynamics = torch.squeeze(self.dynamic['time-reduce'](dynamics), 1)
        src = dynamics.unsqueeze(1).expand([b, n, n, f])
        dst = dynamics.unsqueeze(2).expand([b, n, n, f])
        adj_mxs = self.dynamic['arcs'](torch.cat([src, dst], -1)).permute([3, 0, 1, 2])  # [edge_dim, B, N, N]

        identity = torch.eye(n, dtype=torch.float32, device=dynamics.device)
        adj_mxs = F.tanh(adj_mxs.contiguous()) * 2 / math.pi
        dynamic_supports = torch.max(adj_mxs, identity) - identity

        return dynamic_supports


def testing():
    config = {
        'dynamic_bias': None,
        'num_node': 207,
        'n_in': 2,
        'n_out': 1,
        'n_hist': 12,
        'n_pred': 12,
        'node_dim': 8,
        'edge_dim': 2,
        'n_residuals': 8,
        'n_dilations': 8,
        'n_skips': 64,
        'n_ends': 128,
        'kernel_size': 2,
        'n_blocks': 4,
        'n_layers': 2,
        'order': 1,
        'dropout': 0.3
    }
    batch_size = 64
    factor = np.random.randn(config['num_node'], config['node_dim'])
    m = Ours('cpu', factor, **config)

    from utils.utils import get_number_of_parameters

    print(get_number_of_parameters(m))
    x = torch.randn(batch_size, config['n_hist'], config['num_node'], config['n_in'])
    y = m(x)
    print(y.shape)
