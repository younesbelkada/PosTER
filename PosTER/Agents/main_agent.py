import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 

from torch.utils.data import DataLoader
from tqdm import tqdm

from PosTER.Agents.trainer_agent import Trainer
from PosTER.Agents.trainer_agent_ft import Trainer_FT

class Trainer_Agent(object):
  """
    Trainer object to monitor training and validation
    :- config -: json config file
  """
  def __init__(self, config):
    self.config = config
    if self.config['General']['Task'].lower() == 'pose-modeling':
      self.trainer = Trainer(config)
    elif self.config['General']['Task'].lower() == 'attribute-classification':
      self.trainer = Trainer_FT(config)
    else:
      raise "Not implemented!"