import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from PosTER.loss import point_loss
from PosTER.dataset import DynamicDataset, StaticDataset, my_collate

def get_optimizer(model, config):
    """
        Get the optimizer according to the one specified on the config file
        :- model -: model used for training
        :- config -: json config file
    """
    optim_type = config['Training']['optimizer']
    if optim_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = config['Training']['learning_rate'])
    elif optim_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = config['Training']['learning_rate'])
    else:
        raise ValueError("Not implemented for the optimizer type {}".format(optim_type))
    return optimizer

def get_criterion(config):
    """
        Get the criterion according to the one specified on the config file
        :- config -: json config file
    """
    criterion_type = config['Training']['criterion']
    if criterion_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif criterion_type.lower() == 'pointloss':
        criterion = point_loss
    elif criterion_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif criterion_type.lower() == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Not implemented for the criterion type {}".format(criterion_type))
    return criterion

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_dataset(config):
    """
        Get the dataset according to the config file
        :- config -: json config file that specifies the details about
        which dataset to load
    """
    dataset_type = config['General']['DatasetType']
    if dataset_type.lower() == 'static':
        train_data = StaticDataset(config, 'train')
        train_dataloader = DataLoader(train_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate, shuffle=True)

        val_data = StaticDataset(config, 'val')
        val_dataloader = DataLoader(val_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate)
    elif dataset_type.lower() == 'dynamic':
        train_data = DynamicDataset(config, 'train')
        train_dataloader = DataLoader(train_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate, shuffle=True)

        val_data = DynamicDataset(config, 'val')
        val_dataloader = DataLoader(val_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate)
    return train_dataloader, val_dataloader