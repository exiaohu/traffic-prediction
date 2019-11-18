from typing import Tuple

import torch
from torch import Tensor, nn

from networks.dcrnn import DCRNN
from networks.stgcn import STGCN
from utils import get_graph, scaled_laplacian, cheb_poly_approx


def create_model(name: str, dataset: str, loss, config: dict, device):
    if name == 'STGCN':
        return STGCN(**config)
    elif name == 'DCRNN':
        model = DCRNN(**config)
        graph = scaled_laplacian(get_graph(dataset), lambda_max=None)
        graph = cheb_poly_approx(graph.todense(), config['k_hop'], graph.shape[0])
        graph = torch.tensor(graph, dtype=torch.float32, device=device)
        return model, DCRNNTrainer(model, loss, graph)
    else:
        raise ValueError(f'{name} is not implemented.')


class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class DCRNNTrainer(Trainer):
    def __init__(self, model: DCRNN, loss, graph_filters: Tensor):
        super(DCRNNTrainer, self).__init__(model, loss)
        self.graph_filters = graph_filters
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += inputs.shape[0]
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, self.graph_filters, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets)
        return outputs, loss
