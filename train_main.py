import argparse
import json
import os
import shutil

import torch
import yaml

from models import create_model
from utils import train_model, get_optimizer, get_loss, get_scheduler, test_model, get_datasets


def train(_config, resume: bool = False, test: bool = False):
    print(json.dumps(config, indent=4))

    device = torch.device(_config['device'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    device = torch.device(0)

    dataset = _config['data']['dataset']
    model_name = _config['model']['name']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']

    loss = get_loss(_config['loss']['name'])

    loss.to(device)

    adaptor, model, trainer = create_model(dataset,
                                           loss,
                                           _config['model'][model_name],
                                           _config['trainer'],
                                           device)

    params = [
        {'params': adaptor.parameters(), 'lr': 1e-3, },
        {'params': model.parameters()}
    ]
    optimizer = get_optimizer(optimizer_name, params, **_config['optimizer'][optimizer_name])

    if scheduler_name is not None:
        scheduler = get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])
    else:
        scheduler = None

    save_folder = os.path.join('saves', dataset, _config['name'])

    if not resume and not test:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    datasets = get_datasets(dataset, _config['data']['input_dim'], _config['data']['output_dim'])

    if not test:
        adaptor, model = train_model(
            adaptor=adaptor,
            model=model,
            datasets=datasets,
            batch_size=_config['data']['batch-size'],
            optimizer=optimizer,
            scheduler=scheduler,
            folder=save_folder,
            trainer=trainer,
            device=device,
            **_config['train'])

    test_model(
        adaptor=adaptor,
        model=model,
        datasets=datasets,
        batch_size=_config['data']['batch-size'],
        trainer=trainer,
        folder=save_folder,
        device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--resume', required=False, type=bool, default=False,
                        help='if to resume a trained model?')
    parser.add_argument('--test', required=False, type=bool, default=False,
                        help='if in the test mode?')
    parser.add_argument('--name', required=True, type=str, help='Name.')

    args = parser.parse_args()

    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        config['name'] = args.name
    if args.resume:
        print(f'Resume to {config["name"]}.')
        train(config, resume=True)
    elif args.test:
        print(f'Test {config["name"]}.')
        train(config, test=True)
    else:
        train(config)
