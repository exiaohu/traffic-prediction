import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ZScoreScaler:
    def __init__(self, mean: float, std: float):
        assert std > 0
        self.mean = mean
        self.std = std

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


class TrafficPredictionDataset(Dataset):
    def __init__(self, data_path: str):
        data = np.load(data_path)

        # [num_samples, seq_length, num_nodes, num_features]
        self.inputs: np.ndarray = data['x']
        self.targets: np.ndarray = data['y']
        assert self.inputs.shape[0] == self.targets.shape[0]
        assert self.inputs.shape[2] == self.targets.shape[2]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x).float(), torch.tensor(y[..., :1]).float()

    @property
    def std(self):
        return self.inputs.std(tuple(range(len(self.inputs.shape) - 1)))

    @property
    def mean(self):
        return self.inputs.mean(tuple(range(len(self.inputs.shape) - 1)))


def get_datasets(dataset: str) -> Dict[str, TrafficPredictionDataset]:
    return {key: TrafficPredictionDataset(os.path.join('data', dataset, f'{key}.npz')) for key in
            ['train', 'val', 'test']}


def get_dataloaders(datasets: Dict[str, TrafficPredictionDataset],
                    batch_size: int,
                    num_workers: int = 16) -> Dict[str, DataLoader]:
    return {key: DataLoader(dataset=ds,
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers) for key, ds in datasets.items()}
