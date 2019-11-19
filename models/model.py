from typing import Tuple

from torch import Tensor, nn

from networks.dcrnn import DCRNN
from networks.fc_lstm import FCLSTM
from networks.stgcn import STGCN
from utils import get_graph, scaled_laplacian, convert_scipy_to_torch_sparse


def create_model(name: str, dataset: str, loss, config: dict, device):
    if name == 'STGCN':
        return STGCN(**config)
    elif name == 'DCRNN':
        model = DCRNN(**config)
        graph = scaled_laplacian(get_graph(dataset), lambda_max=None)
        graph = convert_scipy_to_torch_sparse(graph).to(device)
        return model, DCRNNTrainer(model, loss, graph)
    elif name == 'FCLSTM':
        model = FCLSTM(**config)
        return model, FCLSTMTrainer(model, loss)
    else:
        raise ValueError(f'{name} is not implemented.')


class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class DCRNNTrainer(Trainer):
    def __init__(self, model: DCRNN, loss, graph: Tensor):
        super(DCRNNTrainer, self).__init__(model, loss)
        self.graph = graph
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs = self.model(inputs, [self.graph], i_targets, self.train_batch_seen)
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
