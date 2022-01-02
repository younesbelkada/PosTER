import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from PosTER.loss import point_loss, pose_bt_loss, MultiTaskLossWrapper
from PosTER.dataset import  StaticDataset, my_collate, DynamicDataset, StaticDataset
from PosTER.TITAN.titan_dataset import TITANDataset, TITANSimpleDataset
from PosTER.TCG.tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_collate_fn, tcg_pad_seqs

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
    elif optim_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = config['Training']['learning_rate'])
    else:
        raise ValueError("Not implemented for the optimizer type {}".format(optim_type))
    return optimizer

def get_criterion(config):
    """
        Get the criterion according to the one specified on the config file
        :- config -: json config file
    """
    criterion_type = config['Training']['criterion']['type']
    if criterion_type.lower() == 'mse':
        criterion = pose_bt_loss
    elif criterion_type.lower() == 'pointloss':
        criterion = point_loss
    elif criterion_type.lower() == 'mae':
        criterion = nn.L1Loss()
    elif criterion_type.lower() == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
        #criterion = F.cross_entropy
        #criterion = MultiTaskLossWrapper()
    else:
        raise ValueError("Not implemented for the criterion type {}".format(criterion_type))
    return criterion

def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])


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
    elif dataset_type.lower() == 'titan':
        train_data = TITANDataset(pickle_dir=config['Dataset']['TITAN']['pickle_dir'], split='train', dataset_dir=config['Dataset']['TITAN']['dataset_dir'])
        train_simple_dataset = TITANSimpleDataset(train_data, merge_cls=False)
        train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True, collate_fn=TITANSimpleDataset.collate)
        #train_dataloader = DataLoader(train_data, batch_size=config['Training']['batch_size'], shuffle=True)

        val_data = TITANDataset(pickle_dir=config['Dataset']['TITAN']['pickle_dir'], split='val',  dataset_dir=config['Dataset']['TITAN']['dataset_dir'])
        
        val_simple_dataset = TITANSimpleDataset(val_data, merge_cls=False)
        val_dataloader = DataLoader(val_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True, collate_fn=TITANSimpleDataset.collate)
        #val_dataloader = DataLoader(val_data, batch_size=config['Training']['batch_size'])
    elif dataset_type.lower() == 'tcg':
        datapath, label_type = config['Dataset']['TCG']['dataset_dir'], config['Dataset']['TCG']['label_type']
        train_data = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
        train_simple_dataset = TCGSingleFrameDataset(train_data)
        train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
        
        val_data = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=False)
        val_simple_dataset = TCGSingleFrameDataset(val_data)
        val_dataloader = DataLoader(val_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
        

    return train_dataloader, val_dataloader