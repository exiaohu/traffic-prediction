from typing import Tuple, List

from torch import nn, Tensor

from networks.dcrnn import DCRNN
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
