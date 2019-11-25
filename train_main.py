import argparse
import json
import os
import shutil

import torch
import yaml

from models import create_model
from utils import train_model, get_optimizer, get_loss, get_scheduler


def resume(_config: dict):
    print(json.dumps(config, indent=4))
    dataset = _config['data']['dataset']
    model_name = _config['model']['name']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']

    loss = get_loss(_config['loss']['name'])

    loss.to(_config['train']['device'])

    model, trainer = create_model(model_name,
                                  dataset,
                                  get_loss(_config['loss']['name']),
                                  _config['model'][model_name],
                                  _config['train']['device'])

    optimizer = get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])

    scheduler = get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])

    save_folder = os.path.join('saves', dataset, _config['name'])

    saved = torch.load(os.path.join(save_folder, 'best_model.pkl'))

    model.load_state_dict(saved['model_state_dict'])
    begin_epoch = saved['epoch'] + 1
    optimizer.load_state_dict(saved['optimizer_state_dict'])

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    train_model(model=model,
                dataset=dataset,
                batch_size=_config['data']['batch-size'],
                optimizer=optimizer,
                scheduler=scheduler,
                folder=save_folder,
                trainer=trainer,
                begin_epoch=begin_epoch,
                **_config['train'])


def train(_config):
    print(json.dumps(config, indent=4))
    dataset = _config['data']['dataset']
    model_name = _config['model']['name']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']

    loss = get_loss(_config['loss']['name'])

    loss.to(_config['train']['device'])

    model, trainer = create_model(model_name,
                                  dataset,
                                  loss,
                                  _config['model'][model_name],
                                  _config['train']['device'])

    optimizer = get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])

    scheduler = get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])

    save_folder = os.path.join('saves', dataset, _config['name'])

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    train_model(model=model,
                dataset=dataset,
                batch_size=_config['data']['batch-size'],
                optimizer=optimizer,
                scheduler=scheduler,
                folder=save_folder,
                trainer=trainer,
                begin_epoch=0,
                **_config['train'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--resume', required=False, type=bool, default=False,
                        help='Resume.')

    args = parser.parse_args()

    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
    if args.resume:
        print(f'Resume to {config["name"]}.')
        resume(config)
    else:
        train(config)
