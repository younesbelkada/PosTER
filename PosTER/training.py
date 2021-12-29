import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from PosTER.model import PosTER, PosTER_FT, PredictionHeads
from PosTER.utils_train import save_checkpoint, load_checkpoint, get_optimizer, get_criterion
from PosTER.utils import convert_xyc_numpy
from PosTER.augmentations import RandomMask, BodyParts
from PosTER.paint_keypoints import KeypointPainter, COCO_PERSON_SKELETON

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

class Trainer(object):
  """
    Trainer object to monitor training and validation
    :- config -: json config file
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
    log_interval = self.config['Training']['log_interval']
    intermediate_dist_loss = 0
    intermediate_bt_loss = 0

    training_samples_to_plot = []
    for batch_idx, input_batch in enumerate(loop):
      if self.config['General']['Task'] == "Pose-modeling":
        masked_keypoints, full_keypoints = self.mask_transform(input_batch)
        masked_keypoints, full_keypoints = masked_keypoints.to(self.device), full_keypoints.to(self.device)
        full_keypoints = torch.flatten(full_keypoints, start_dim=1)
        if len(training_samples_to_plot) < self.config['Training']['n_samples_visualization']:
          training_samples_to_plot.append((torch.flatten(masked_keypoints.detach().cpu(), start_dim=1)[0, :], full_keypoints[0, :].detach().cpu(), masked_keypoints[0, :].detach().cpu()))
        
        cls_tokens_masked, predicted_keypoints = self.model(masked_keypoints)
        cls_tokens_full, _ = self.model(BodyParts()(full_keypoints))
        
        dist_loss, bt_loss = self.criterion(predicted_keypoints, full_keypoints, cls_tokens_masked, cls_tokens_full, lmbda=self.config['Training']['criterion']['lmbda'], enable_bt=self.config['Training']['criterion']['enable_bt'])
        loss = dist_loss
        if bt_loss:
          loss = (dist_loss + bt_loss)/2
      else:
        raise BaseException("task {} not implemented for train".format(self.config['General']['Task']))
            
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      avg_loss += loss.item()
      intermediate_dist_loss += dist_loss.item()
      if bt_loss:
        intermediate_bt_loss += bt_loss.item()
      
      if batch_idx%log_interval == 0:
        if self.config['wandb']['enable']:
          wandb.log({
              "intermediate_loss": intermediate_dist_loss/log_interval,
              "intermediate_bt_loss": intermediate_bt_loss/log_interval
          })
        intermediate_dist_loss = 0
        intermediate_bt_loss = 0
    avg_loss = avg_loss/len(train_loader)
    

    # Evaluate model on the validation set
    avg_val_loss = 0
    validation_samples_to_plot = []
    with torch.no_grad():
      self.model.eval()
      for batch_idx, input_batch in enumerate(tqdm(val_loader)):
        if self.config['General']['Task'] == "Pose-modeling":
          masked_keypoints, full_keypoints = self.mask_transform(input_batch)
          masked_keypoints, full_keypoints = masked_keypoints.to(self.device), full_keypoints.to(self.device)
          full_keypoints = torch.flatten(full_keypoints, start_dim=1)
          if len(validation_samples_to_plot) < self.config['Training']['n_samples_visualization']:
            validation_samples_to_plot.append((torch.flatten(masked_keypoints.detach().cpu(), start_dim=1)[0, :], full_keypoints[0, :].detach().cpu(), masked_keypoints[0, :].detach().cpu()))
          cls_tokens_masked, predicted_keypoints = self.model(masked_keypoints)
          cls_tokens_full, _ = self.model(BodyParts()(full_keypoints))
          dist_loss_val, bt_loss_val = self.criterion(predicted_keypoints, full_keypoints, cls_tokens_masked, cls_tokens_full, lmbda=self.config['Training']['criterion']['lmbda'], enable_bt=self.config['Training']['criterion']['enable_bt'])
          val_loss = dist_loss_val
          if bt_loss_val:
            val_loss = (dist_loss_val + bt_loss_val)/2
          avg_val_loss += val_loss.item()        
        else:
          raise "task {} not implemented in val".format(self.config['General']['Task'])
        
    avg_val_loss = avg_val_loss/len(val_loader)
    if self.config['wandb']['enable']:
      self.show_comparison(training_samples_to_plot, validation_samples_to_plot)
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
            save_checkpoint(self.model)
  def show_comparison(self, training_samples, validation_samples):
    """
      Runs an inference on some samples in the validation set and 
      compares the result between the masked input and the predicted 
      output. 
    """

    print("Generating some samples on the training set")
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=self.config['Training']['n_samples_visualization'], figsize=(10,10))
    fig.suptitle("Example predictions on the training set", fontsize=20)
    kps_painter = KeypointPainter()
    with torch.no_grad():
      for i, training_sample in enumerate(training_samples):
        plt.axis('off')
        x, y, v = convert_xyc_numpy(training_sample[1].numpy())
        masked_x, masked_y, _ = convert_xyc_numpy(training_sample[0].numpy())
        axes[0, i].invert_yaxis()
        axes[0, i].axis('off')
        kps_painter._draw_skeleton(axes[0, i], x, y, v, skeleton=COCO_PERSON_SKELETON, masked_x=masked_x, masked_y=masked_y, mask_joints=True)
      
        _ , predicted_keypoints = self.model(training_sample[-1].to(self.device).unsqueeze(0))
        predicted_x, predicted_y, predicted_v = convert_xyc_numpy(predicted_keypoints.squeeze(0).detach().cpu().numpy())
        axes[1, i].invert_yaxis()
        axes[1, i].axis('off')
        kps_painter._draw_skeleton(axes[1, i], predicted_x, predicted_y, predicted_v, skeleton=COCO_PERSON_SKELETON, mask_joints=False)
    #plt.show()
    if self.config['wandb']['enable']:
      plot = wandb.Image(plt)
      wandb.log(
        {
          "training_plot":plot
        }
      )
    
    print("Generating some samples on the training set")
    plt.ioff()
    fig, axes = plt.subplots(nrows=2, ncols=self.config['Training']['n_samples_visualization'], figsize=(10,10))
    fig.suptitle("Example predictions on the validation set", fontsize=20)
    kps_painter = KeypointPainter()
    with torch.no_grad():
      for i, val_sample in enumerate(validation_samples):
        plt.axis('off')
        x, y, v = convert_xyc_numpy(val_sample[1].numpy())
        masked_x, masked_y, _ = convert_xyc_numpy(val_sample[0].numpy())
        axes[0, i].invert_yaxis()
        axes[0, i].axis('off')
        kps_painter._draw_skeleton(axes[0, i], x, y, v, skeleton=COCO_PERSON_SKELETON, masked_x=masked_x, masked_y=masked_y, mask_joints=True)
      
        _, predicted_keypoints = self.model(val_sample[-1].to(self.device).unsqueeze(0))
        predicted_x, predicted_y, predicted_v = convert_xyc_numpy(predicted_keypoints.squeeze(0).detach().cpu().numpy())
        axes[1, i].invert_yaxis()
        axes[1, i].axis('off')
        kps_painter._draw_skeleton(axes[1, i], predicted_x, predicted_y, predicted_v, skeleton=COCO_PERSON_SKELETON, mask_joints=False)
    #plt.show()
    if self.config['wandb']['enable']:
      plot = wandb.Image(plt)
      wandb.log(
        {
          "validation_plot":plot
        }
      )
    plt.close()


class Trainer_FT(object):
  """
    Trainer object to monitor training and validation
    :- config -: json config file
  """
  def __init__(self, config):
    self.config = config
    if config['General']['DatasetType'] == 'TITAN':
      attributes = [4, 7, 9, 13, 4]
    else:
      raise "Not implemented"
    self.heads = PredictionHeads(attributes)
    print(self.heads)
    self.poster_model = PosTER(config)
    checkpoint_file = "my_checkpoint.pth.tar"
    #load_checkpoint(checkpoint_file, self.poster_model)
    self.model = PosTER_FT(self.poster_model , self.heads)
    self.optimizer = get_optimizer(self.model, config)
    self.criterion = get_criterion(config)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.criterion = self.criterion.to(self.device)
  def train_one_epoch(self, train_loader, val_loader):
    """
      Train the model according to the args specified on the config file and 
      given the train and validation dataloader for only one epoch
      :- train_loader -: training dataloader
      :- val_loader -: validation dataloader
    """
    torch.autograd.set_detect_anomaly(True)
    self.model.train()
    loop = tqdm(train_loader)
    avg_loss = 0
    log_interval = self.config['Training']['log_interval']
    intermediate_loss = 0

    for batch_idx, input_batch in enumerate(loop):
      keypoints, attributes = input_batch 
      keypoints, attributes = keypoints.to(self.device), attributes.to(self.device)
      
      predictions = self.model(keypoints)
      loss = 0
      for i, pred_tensors in enumerate(predictions):
        targets = attributes[:, i].detach().cpu().clone()
        loss += self.criterion(pred_tensors, targets.to(self.device))
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      
      avg_loss += loss.item()
      intermediate_loss += loss.item()
      if batch_idx%log_interval == 0:
        if self.config['wandb']['enable']:
          wandb.log({
              "intermediate_loss": intermediate_loss/log_interval
          })
        intermediate_loss = 0
    avg_loss = avg_loss/len(train_loader)
    

    # Evaluate model on the validation set
    avg_val_loss = 0
    validation_samples_to_plot = []
    with torch.no_grad():
      self.model.eval()
      for batch_idx, input_batch in enumerate(tqdm(val_loader)):
        keypoints, attributes = input_batch
        keypoints, attributes = keypoints.to(self.device), attributes.to(self.device)
        
        predictions = self.model(keypoints)
        loss = 0
        for i, pred_tensors in enumerate(predictions):
          targets = attributes[:, i].detach().cpu().clone()
          loss += self.criterion(pred_tensors, targets.to(self.device))
        avg_val_loss += loss.item()
    avg_val_loss = avg_val_loss/len(val_loader)
    #if self.config['wandb']['enable']:
    #  self.show_comparison(training_samples_to_plot, validation_samples_to_plot)
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
            save_checkpoint(self.model)