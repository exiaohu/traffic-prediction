import argparse
import os
import shutil

import yaml

from models import create_model
from utils import train_model, get_dataloaders, get_optimizer, get_loss, get_graph, get_scheduler
from utils.data import ZScoreScaler
from utils.graph import scaled_laplacian, cheb_poly_approx


def train(_config: dict):
    dataset = _config['data']['dataset']
    model_name = _config['model']['name']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['train']['scheduler']

    model = create_model(_config['model'][model_name])
    mean, std = _config['data'][dataset]['mean'], _config['data'][dataset]['std']
    scaler = ZScoreScaler(mean, std)
    dls = get_dataloaders(dataset, _config['data']['batch-size'])

    loss = get_loss(_config['loss']['name'])

    optimizer = get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])

    scheduler = get_scheduler(scheduler_name, optimizer, **_config['train'][scheduler_name])

    graph = get_graph(dataset)
    graph = cheb_poly_approx(scaled_laplacian(graph), _config['model'][model_name]['k_hop'], graph.shape[0])

    save_folder = os.path.join('saves', dataset, _config['name'])

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    train_model(model=model,
                dataloaders=dls,
                criterion=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                graph=graph,
                folder=save_folder,
                scaler=scaler,
                **_config['train'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='Configuration filename for restoring the model.')

    args = parser.parse_args()

    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
    train(config)
