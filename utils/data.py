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
        self.inputs, self.targets = data['x'], data['y']
        assert self.inputs.shape[0] == self.targets.shape[0]
        assert self.inputs.shape[2] == self.targets.shape[2]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x[..., :1]).float(), torch.tensor(y[..., :1]).float()


def get_dataloaders(dataset: str, batch_size: int, num_workers: int = 16) -> Dict[str, DataLoader]:
    return {key: DataLoader(TrafficPredictionDataset(os.path.join('data', dataset, f'{key}.npz')),
                            batch_size=batch_size,
                            shuffle=(key == 'train'),
                            num_workers=num_workers)
            for key in ['train', 'val', 'test']}
