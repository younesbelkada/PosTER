import wandb
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from PosTER.dataset import DynamicDataset, StaticDataset
from PosTER.model import PosTER 
from PosTER.utils_train import save_checkpoint, load_checkpoint, get_optimizer, get_criterion
from PosTER.augmentations import RandomMask

class Trainer(object):
  """
    Trainer object to monitor training and validation
    :- model -: model to use for training 
  """
  def __init__(self, config):
    self.config = config
    self.model = PosTER(config)
    self.optimizer = get_optimizer(self.model, config)
    self.criterion = get_criterion(config)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.config['General']['Task'] == "Pose-modeling":
      self.mask_transform = RandomMask(self.config['Dataset']['Transforms']['mask']['N'])
    
  def train_one_epoch(self, train_loader, val_loader):
    """
      Train the model according to the args specified on the config file and 
      given the train and validation dataloader for only one epoch
      :- train_loader -: training dataloader
      :- val_loader -: validation dataloader
    """
    self.model.train()
    loop = tqdm(train_loader)
    avg_loss = 0
    for batch_idx, input_batch in enumerate(loop):
      if self.config['General']['Task'] == "Pose-modeling":
        masked_keypoints, full_keypoints = self.mask_transform(input_batch)
        masked_keypoints, full_keypoints = masked_keypoints.to(self.device), full_keypoints.to(self.device)
        full_keypoints = torch.flatten(full_keypoints, start_dim=1)
      else:
        raise "task {} not implemented for train".format(self.config['General']['Task'])
            
      predicted_keypoints = self.model(masked_keypoints)
      loss = self.criterion(predicted_keypoints, full_keypoints)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      avg_loss += loss.item()
    avg_loss = avg_loss/len(train_loader)
    

    # Evaluate model on the validation set
    avg_val_loss = 0
    with torch.no_grad():
      self.model.eval()
      for batch_idx, input_batch in enumerate(tqdm(val_loader)):
        if self.config['General']['Task'] == "Pose-modeling":
          masked_keypoints, full_keypoints = self.mask_transform(input_batch)
          masked_keypoints, full_keypoints = masked_keypoints.to(self.device), full_keypoints.to(self.device)
          full_keypoints = torch.flatten(full_keypoints, start_dim=1)
        else:
          raise "task {} not implementedin val".format(self.config['General']['Task'])

        predicted_keypoints = self.model(masked_keypoints)
        loss_val = self.criterion(predicted_keypoints, full_keypoints)
          
        avg_val_loss += loss_val.item()

    avg_val_loss = avg_val_loss/len(val_loader)
    return avg_loss, avg_val_loss

  def train(self, train_loader, val_loader):
    """
      Train the model according to the args specified on the config file and 
      given the train and validation dataloader
      :- train_loader -: training dataloader
      :- val_loader -: validation dataloader
    """

    if self.config['wandb']['enable']:
      wandb_entity = self.config['wandb']['entity']
      wandb.init(project=self.config['wandb']['project_name'], entity=wandb_entity)
      wandb.watch(self.model, self.criterion, log="all", log_freq=10)

    best_loss = float('inf')
    self.model = self.model.to(self.device)
    for epoch in range(self.config['Training']['epochs']):
        #Train epoch and return losses
        loss, val_loss = self.train_one_epoch(train_loader, val_loader)

        #Display results
        print(f"Loss epoch {epoch}: ", loss)
        print(f"Validation Loss epoch {epoch}: ", val_loss)

        #Log results in wandb
        if self.config['wandb']['enable']:
          wandb.log({
              "loss": loss, 
              "epoch": epoch,
              "val_loss": val_loss
          })
        
        #Save best model
        if (self.config['Training']['save_checkpoint'] and val_loss < best_loss):
            best_loss = val_loss
            save_checkpoint(self.model, self.optimizer)    