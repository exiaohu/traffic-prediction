import math
from typing import Tuple, List

from torch import nn, Tensor

from networks.dcrnn import DCRNN
from networks.dcrnn_model import DCRNNModel
from networks.fc_lstm import FCLSTM
from networks.ours import Ours
from networks.stgcn import STGCN
from utils import get_graph, scaled_laplacian, convert_scipy_to_torch_sparse


class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


def create_model(name: str, dataset: str, loss, config: dict, device) -> Tuple[nn.Module, Trainer]:
    if name == 'STGCN':
        model = STGCN(**config)
        return model, STGCNTrainer(model, loss)
    elif name == 'DCRNN':
        model = DCRNN(**config)
        graph = get_graph(dataset)
        graph = scaled_laplacian(graph)
        graph = convert_scipy_to_torch_sparse(graph).to(device)
        # g1, g2 = random_walk_matrix(graph), reverse_random_walk_matrix(graph)
        # g1, g2 = convert_scipy_to_torch_sparse(g1).to(device), convert_scipy_to_torch_sparse(g2).to(device)
        return model, DCRNNTrainer(model, loss, [graph])
    elif name == 'FCLSTM':
        model = FCLSTM(**config)
        return model, FCLSTMTrainer(model, loss)
    elif name == 'Ours':
        model = Ours(**config)
        return model, OursTrainer(model, loss)
    elif name == 'DCRNNMODEL':
        cl_decay_steps = config.pop('cl_decay_steps')
        graph = get_graph(dataset)
        model = DCRNNModel(graph, **config)
        return model, DCRNNMODELTrainer(model, loss, cl_decay_steps)
    else:
        raise ValueError(f'{name} is not implemented.')


class STGCNTrainer(Trainer):
    pass


class DCRNNTrainer(Trainer):
    def __init__(self, model: DCRNN, loss, graphs: List[Tensor]):
        super(DCRNNTrainer, self).__init__(model, loss)
        for graph in graphs:
            graph.requires_grad_(False)
        self.graphs = graphs
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, self.graphs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss


class FCLSTMTrainer(Trainer):
    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        if phase == 'train':
            outputs = self.model(inputs, targets)
        else:
            outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        return outputs, loss


class OursTrainer(Trainer):
    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        if phase == 'train':
            outputs, supports = self.model(inputs, targets)
        else:
            outputs, supports = self.model(inputs)
        loss = self.loss(outputs, targets)  # - 100 * distributions.Categorical(probs=supports).entropy().mean()
        return outputs, loss


class DCRNNMODELTrainer(Trainer):
    def __init__(self, model: DCRNNModel, loss, cl_decay_steps: int):
        super(DCRNNMODELTrainer, self).__init__(model, loss)
        self.cl_decay_steps = cl_decay_steps
        self.global_step: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        if phase == 'train':
            self.global_step += 1
            outputs = self.model(inputs, targets, self._compute_sampling_threshold)
            outputs = outputs.reshape(12, -1, 207, 1).transpose(0, 1)
        else:
            outputs = self.model(inputs, targets, 0)
            outputs = outputs.reshape(12, -1, 207, 1).transpose(0, 1)
        loss = self.loss(outputs, targets)
        return outputs, loss

    @property
    def _compute_sampling_threshold(self):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_step / self.cl_decay_steps))
