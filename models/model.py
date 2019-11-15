from networks.dcrnn import DCRNN
from networks.stgcn import STGCN


def create_model(name: str, config: dict):
    if name == 'STGCN':
        return STGCN(**config)
    elif name == 'DCRNN':
        return DCRNN(**config)
    else:
        raise ValueError(f'{name} is not implemented.')
