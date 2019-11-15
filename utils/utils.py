import copy
import os
import pickle
import time
from typing import Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.dcrnn import DCRNN
from networks.stgcn import STGCN
from .evaluate import evaluate


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_loss(name, **kwargs):
    return getattr(nn, name)(**kwargs)


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


def train_model(model: nn.Module,
                dataloaders: Dict[str, DataLoader],
                criterion: nn.Module,
                optimizer,
                scheduler,
                graph: np.ndarray,
                folder: str,
                scaler,
                epochs: int,
                device: str):
    phases = ['train', 'val', 'test']

    writer = SummaryWriter(folder)

    since = time.perf_counter()

    if isinstance(model, STGCN):
        model = MultiStepWraper(model)

    model = model.to(device)
    criterion = criterion.to(device)

    save_dict, best_criterion = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 0

    graph = torch.tensor(graph, device=device, dtype=torch.float32)

    try:
        for epoch in range(epochs):

            running_loss, running_metrics = {phase: 0.0 for phase in phases}, {phase: dict() for phase in phases}
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    inputs = scaler.transform(inputs).to(device)
                    targets = scaler.transform(targets).to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        if isinstance(model, MultiStepWraper):
                            outputs = model(inputs, graph, targets, (phase == 'train'))
                        elif isinstance(model, DCRNN):
                            i_targets = targets if phase == 'train' else None
                            batch_seen = step + epoch * (
                                    len(dataloaders['train'].dataset) // dataloaders['train'].batch_size)
                            outputs = model(inputs, graph, i_targets, batch_seen)

                        loss = criterion(outputs, targets)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(scaler.inverse_transform(outputs.detach().cpu().numpy()))
                        running_targets.append(scaler.inverse_transform(targets.detach().cpu().numpy()))

                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                # 性能
                scores = evaluate(np.concatenate(running_targets), np.concatenate(predictions))
                running_metrics[phase] = scores

                if phase == 'validate' and scores['F1-SCORE'] > best_criterion:
                    best_criterion = scores['F1-SCORE'],
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                     epoch=epoch,
                                     optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                writer.add_scalars(metric, {
                    f'{phase} {metric}': running_metrics[phase][metric] for phase in phases},
                                   global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    finally:
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        model.load_state_dict(save_dict['model_state_dict'])

        save_model(os.path.join(folder, 'best_model.pkl'), **save_dict)

    return model


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
                targets: torch.Tensor,
                is_train: bool):
        """
        generate multi-step predictions from single-step predicting method
        :param inputs: tensor, [B, N_hist, N, F_in]
        :param graph_filters: tensor, [N, K_hop, N]
        :param targets: tensor, multi-step prediction targets, [B, N_pred, N, F_out]
        :param is_train: whether on train mode or not
        :return: tensor, with the same shape as `targets`
        """
        _, n_hist, _, _ = inputs.shape
        _, n_pred, _, _ = targets.shape
        h, preds = inputs.clone(), list()
        for i in range(n_pred):
            prediction = self.model(h[:, i:i + n_hist, ...], graph_filters)
            preds.append(prediction)
            if not is_train:
                h = torch.cat([h, prediction.unsqueeze(1)], 1)
            else:
                h = torch.cat([h, targets[:, i:i + 1]], 1)
                h.requires_grad_()
        return torch.stack(preds, 1)
