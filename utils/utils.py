import copy
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import get_datasets, get_dataloaders
from .evaluate import evaluate


def train_model(model: nn.Module,
                datasets: Dict[str, Dataset],
                batch_size: int,
                optimizer,
                scheduler,
                folder: str,
                trainer,
                epochs: int,
                device,
                max_grad_norm: float = None):
    dataloaders = get_dataloaders(datasets, batch_size)
    # scaler = ZScoreScaler(datasets['train'].mean[0], datasets['train'].std[0])

    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'val', 'test']

    writer = SummaryWriter(folder)

    since = time.perf_counter()

    model = model.to(device)

    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        # inputs[..., 0] = scaler.transform(inputs[..., 0])
                        inputs = inputs.to(device)
                        # targets[..., 0] = scaler.transform(targets[..., 0])
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        # predictions.append(scaler.inverse_transform(outputs).cpu().numpy())
                        predictions.append(outputs.cpu().numpy())

                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                # 性能
                running_metrics[phase] = evaluate(np.concatenate(predictions), np.concatenate(running_targets))

                if phase == 'val':
                    if running_loss['val'] < best_val_loss:
                        best_val_loss = running_loss['val']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > 10:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        model.load_state_dict(save_dict['model_state_dict'])

        save_model(save_path, **save_dict)
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    return model


def test_model(model: nn.Module,
               datasets: Dict[str, Dataset],
               batch_size: int,
               trainer,
               folder: str,
               device):
    dataloaders = get_dataloaders(datasets, batch_size)
    # scaler = ZScoreScaler(datasets['train'].mean[0], datasets['train'].std[0])

    save_path = os.path.join(folder, 'best_model.pkl')

    save_dict = torch.load(save_path)

    model.load_state_dict(save_dict['model_state_dict'])

    if isinstance(device, str):
        model = model.to(device)
    elif isinstance(device, list):
        model = nn.DataParallel(model, device)
        device = f'cuda:{device[0]}'

    model.eval()

    predictions, running_targets = list(), list()
    for step, (inputs, targets) in tqdm(enumerate(dataloaders['test']), 'Test model'):
        with torch.no_grad():
            running_targets.append(targets.numpy())

            # inputs[..., 0] = scaler.transform(inputs[..., 0])
            inputs = inputs.to(device)
            # targets[..., 0] = scaler.transform(targets[..., 0])
            targets = targets.to(device)

            outputs, _ = trainer.train(inputs, targets, 'test')
            # predictions.append(scaler.inverse_transform(outputs).cpu().numpy())
            predictions.append(outputs.cpu().numpy())

    # 性能
    predictions, running_targets = np.concatenate(predictions), np.concatenate(running_targets)
    scores = evaluate(predictions, running_targets)
    print('test results:')
    print(json.dumps(scores, cls=JsonEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=JsonEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


def get_graph(dataset: str):
    _, _, g = load_graph_data(os.path.join('data', dataset, 'adj_mx.pkl'))
    return g


class MultiStepWraper(nn.Module):

    def __init__(self, model: nn.Module):
        """
        wrap a model to make multi-step predictions.
        :param model: map inputs to a single predicting result
        """
        super(MultiStepWraper, self).__init__()
        self.model = model

    def forward(self,
                inputs: torch.Tensor,
                graph_filters: torch.Tensor,
                n_pred: int,
                targets: torch.Tensor = None):
        """
        generate multi-step predictions from single-step predicting method
        :param inputs: tensor, [B, N_hist, N, F_in]
        :param graph_filters: tensor, [N, K_hop, N]
        :param n_pred: the number of predictions
        :param targets: tensor, multi-step prediction targets, [B, N_pred, N, F_out]
        :return: tensor, with the same shape as `targets`
        """
        _, n_hist, _, _ = inputs.shape
        h, preds = inputs.clone(), list()
        for i in range(n_pred):
            prediction = self.model(h[:, i:i + n_hist], graph_filters)
            preds.append(prediction)
            if targets is None:
                h = torch.cat([h, prediction.unsqueeze(1)], 1)
            else:
                h = torch.cat([h, targets[:, i:i + 1]], 1)
                h.requires_grad_()
        return torch.stack(preds, 1)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_optimizer(name: str, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
