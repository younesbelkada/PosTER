import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PosTER.loss import point_loss, pose_bt_loss, MultiTaskLossWrapper, pose_bt_loss_mae

from PosTER.Datasets.pie_dataset import  StaticDataset, my_collate, DynamicDataset, StaticDataset
from PosTER.Datasets.titan_dataset import TITANDataset, TITANSimpleDataset
from PosTER.Datasets.tcg_dataset import TCGDataset, TCGSingleFrameDataset, tcg_collate_fn, tcg_pad_seqs
from PosTER.Datasets.utils import from_stats_to_weights, return_weights
from PosTER.Datasets.transforms_agent import TransformsAgent

from PosTER.Models.PosTER import PosTER
from PosTER.Models.PosTER_FT import PosTER_FT
from PosTER.Models.utils_models import PredictionHeads
from PosTER.Models.MonoLoco import MonoLoco

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
        criterion = pose_bt_loss_mae
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

def get_model_for_fine_tuning(config, attributes=None):
    model_type = config['General']['Model_type']
    if model_type == 'PosTER':
        heads = PredictionHeads(attributes)
        poster_model = PosTER(config)
        checkpoint_file = config['Model']['PosTER']['filename']
        load_checkpoint(checkpoint_file, poster_model)
        model = PosTER_FT(poster_model, heads)
    elif model_type == 'MonoLoco':
        assert len(attributes) == 1
        nb_classes = attributes[0]
        model = MonoLoco(51, 0.2, nb_classes)
    return model


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
        merge_cls = config['Dataset']['TITAN']['use_merge']
        train_data = TITANDataset(pickle_dir=config['Dataset']['TITAN']['pickle_dir'], split='train', dataset_dir=config['Dataset']['TITAN']['dataset_dir'])
        transforms = TransformsAgent(config).get_transforms((1980, 1980))
        train_simple_dataset = TITANSimpleDataset(train_data, merge_cls=merge_cls, transforms=transforms, inflate=0.9)
        #train_simple_dataset.all_poses = TITANSimpleDataset.convert_to_relative_coord(train_simple_dataset.all_poses)
        # class_weight = [1/8, 1, 110, 10, 4]
        # samples_weight = []
        # for label in train_simple_dataset.all_labels:
        #     samples_weight.append(class_weight[label[0]])
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        
        #train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=False, collate_fn=TITANSimpleDataset.collate, sampler=sampler)
        #train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=False, collate_fn=TITANSimpleDataset.collate, sampler=sampler)
        train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=False, collate_fn=TITANSimpleDataset.collate)
        #train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
        transforms_val = TransformsAgent(config, test=True).get_transforms((1980, 1980))

        val_data = TITANDataset(pickle_dir=config['Dataset']['TITAN']['pickle_dir'], split='val',  dataset_dir=config['Dataset']['TITAN']['dataset_dir'])
        val_simple_dataset = TITANSimpleDataset(val_data, merge_cls=merge_cls, transforms=transforms_val, inflate=0.0)
        val_dataloader = DataLoader(val_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True, collate_fn=TITANSimpleDataset.collate)
        #val_dataloader = DataLoader(val_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
    elif dataset_type.lower() == 'tcg':
        datapath, label_type = config['Dataset']['TCG']['dataset_dir'], config['Dataset']['TCG']['label_type']
        train_data = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
        train_simple_dataset = TCGSingleFrameDataset(train_data)
        train_dataloader = DataLoader(train_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
        
        val_data = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=False)
        val_simple_dataset = TCGSingleFrameDataset(val_data)
        val_dataloader = DataLoader(val_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=True)
    return train_dataloader, val_dataloader

def get_test_dataset(config):
    """
        Get the test dataset according to the config file
        :- config -: json config file that specifies the details about
        which dataset to load
    """
    dataset_type = config['General']['DatasetType']
    if dataset_type.lower() == 'static':
        test_data = StaticDataset(config, 'test')
        test_dataloader = DataLoader(test_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate, shuffle=False)
    elif dataset_type.lower() == 'dynamic':
        test_data = DynamicDataset(config, 'test')
        test_dataloader = DataLoader(test_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate, shuffle=False)
    elif dataset_type.lower() == 'titan':
        test_data = TITANDataset(pickle_dir=config['Dataset']['TITAN']['pickle_dir'], split='test', dataset_dir=config['Dataset']['TITAN']['dataset_dir'])
        transforms = TransformsAgent(config, test=True).get_transforms((1980, 1980))
        test_simple_dataset = TITANSimpleDataset(test_data, merge_cls=True, transforms=transforms)
        test_dataloader = DataLoader(test_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=False, collate_fn=TITANSimpleDataset.collate)
    elif dataset_type.lower() == 'tcg':
        datapath, label_type = config['Dataset']['TCG']['dataset_dir'], config['Dataset']['TCG']['label_type']
        test_data = TCGDataset(datapath, label_type, eval_type="xs", eval_id=1, training=True)
        test_simple_dataset = TCGSingleFrameDataset(test_data)
        test_dataloader = DataLoader(test_simple_dataset, batch_size=config['Training']['batch_size'], shuffle=False)
    return test_dataloader