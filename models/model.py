import scipy.sparse as sp
import torch
from torch import Tensor, nn

from networks.ours import Ours
from networks.stadaptor import STAdaptor
from utils import load_graph_data


def create_model(dataset: str, model_config: dict, adaptor_config: dict, device):
    supports = load_graph_data(dataset, 'doubletransition')
    supports = torch.tensor(list(map(sp.coo_matrix.toarray, supports)), dtype=torch.float32, device=device)

    edge_dim = supports.size(0)

    adaptor = STAdaptor(supports, **adaptor_config)
    predictor = Ours(edge_dim=edge_dim, **model_config)

    return Model(predictor, adaptor)


class Model(nn.Module):
    def __init__(self, predictor: Ours, adaptor: STAdaptor):
        super(Model, self).__init__()
        self.predictor = predictor
        self.adaptor = adaptor

    def forward(self, inputs: Tensor) -> Tensor:
        supports = self.adaptor(inputs)
        outputs = self.predictor(inputs, supports)
        return outputs
