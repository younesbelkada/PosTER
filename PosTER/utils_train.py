import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from PosTER.loss import point_loss

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
        criterion = point_loss()
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

