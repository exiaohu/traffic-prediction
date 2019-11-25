import sys

import numpy as np
import torch
from torch import nn


def get_loss(name, **kwargs):
    try:
        return getattr(nn, name)(**kwargs)
    except AttributeError:
        try:
            return getattr(sys.modules[__name__], name)(**kwargs)
        except AttributeError:
            raise ValueError(f'{name} is not defined.')


class MaskedMSELoss(nn.Module):
    def __init__(self, null_val=0):
        super(MaskedMSELoss, self).__init__()
        self.null_val = null_val

    def forward(self, preds, labels):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = torch.ne(labels, self.null_val)
        mask = mask.to(torch.float32)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask, device=preds.device), mask)
        loss = torch.pow(preds - labels, 2)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss, device=preds.device), loss)
        return torch.mean(loss)


class MaskedMAELoss(nn.Module):
    def __init__(self, null_val=0):
        super(MaskedMAELoss, self).__init__()
        self.null_val = null_val

    def forward(self, preds, labels):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = torch.ne(labels, self.null_val)
        mask = mask.to(torch.float32)
        mask /= torch.mean(mask)
        print(mask.device, preds.device, labels.device)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask, device=preds.device), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss, device=preds.device), loss)
        return torch.mean(loss)
