from networks.stgcn import STGCN


def create_model(config: dict):
    return STGCN(**config)
