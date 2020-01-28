import shutil
import torch
import yaml
from torch import optim

import models
import utils

learning_rate = 1e-3
weight_decay = 1e-4
adaptor_dropout = 0.5
frozen_predictor = True
node_dim = 3

device = torch.device('cuda:3')

# fixed_model_path = r'data/models/METR-LA/best-fixed.pkl'
# config_path = r'data/models/METR-LA/config.yaml'
fixed_model_path = r'data/models/PEMS-BAY/best-fixed.pkl'
config_path = r'data/models/PEMS-BAY/config.yaml'
saved_folder = r'saves/dynamic'

shutil.rmtree(saved_folder, ignore_errors=True)


with open(config_path) as f:
    config = yaml.safe_load(f)
saved = torch.load(fixed_model_path)

datasets = config['data']['dataset']

model_config = config['model']['Ours1']
adaptor_config = {
    'n_hist': 12,
    'node_dim': node_dim,
    'dropout': adaptor_dropout,
    'spatial': True,
    'temporal': True
}

model = models.create_model(datasets, model_config, adaptor_config, device=device)
model.load_state_dict(saved['model_state_dict'], strict=False)

if frozen_predictor:
    for param in model.predictor.parameters():
        param.requires_grad_(False)

datasets = utils.get_datasets(datasets, 9, 1)
scaler = utils.ZScoreScaler(datasets['train'].mean, datasets['train'].std)
optimizer = optim.Adam([
    {'params': model.adaptor.parameters()},
    {'params': model.predictor.parameters(), 'lr': 1e-5}
], lr=learning_rate)
loss = utils.get_loss('MaskedMAELoss')
trainer = utils.OursTrainer(model, loss, scaler, device, optimizer, weight_decay, 2, 5)

utils.train_model(
    datasets=datasets,
    batch_size=64,
    folder=saved_folder,
    trainer=trainer,
    scheduler=None,
    epochs=100,
    early_stop_steps=10
)

utils.test_model(
    datasets=datasets,
    batch_size=64,
    trainer=trainer,
    folder=saved_folder
)
