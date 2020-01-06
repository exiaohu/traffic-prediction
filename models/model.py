from typing import Tuple

import scipy.sparse as sp
import torch
from torch import Tensor

from networks.ours2 import Ours
from networks.stadaptor import STAdaptor
from utils import load_graph_data


class OursTrainer(object):
    def __init__(self, model: Ours, adaptor: STAdaptor, loss):
        self.model = model
        self.adaptor = adaptor
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        supports = self.adaptor(False, True, inputs)

        outputs = self.model(inputs, supports)
        loss = self.loss(outputs, targets)
        return outputs, loss


def create_model(dataset: str, loss, model_config: dict, trainer_config: dict, device):
    supports = load_graph_data(dataset, 'doubletransition')
    supports = torch.tensor(list(map(sp.coo_matrix.toarray, supports)), dtype=torch.float32, device=device)

    adaptor = STAdaptor(supports, **trainer_config)
    model = Ours(**model_config)

    model.to(device)
    adaptor.to(device)

    return adaptor, model, OursTrainer(model, adaptor, loss)
