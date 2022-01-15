import wandb
import torch
import os

import torch.nn as nn
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np 

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from PosTER.Models.PosTER import PosTER
from PosTER.Models.utils_models import PredictionHeads, BaseLine
from PosTER.Agents.utils_agent import save_checkpoint, get_optimizer, get_criterion, get_model_for_fine_tuning
from PosTER.Datasets.utils import convert_xyc_numpy
from PosTER.Datasets.paint_keypoints import KeypointPainter, COCO_PERSON_SKELETON
from PosTER.Datasets.titan_dataset import Person

class Trainer_FT(object):
  """
    Trainer object to monitor training and validation
    :- config -: json config file
  """
  def __init__(self, config):
    self.config = config
    self.num_attribute_cat = 0
    self.use_merge = config['Dataset']['TITAN']['use_merge']
    if config['General']['DatasetType'] == 'TITAN':
      if self.use_merge:
        n_classes = 5
      else:
        n_classes = 4
      self.num_attribute_cat = 1
    else:
      raise "Not implemented"
    self.model = get_model_for_fine_tuning(config, n_classes)
    self.optimizer = get_optimizer(self.model, config)
    self.criterion = get_criterion(config)
    self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.path_model = os.path.join(self.config['General']['Model_path'], self.model.__class__.__name__+'.p')
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
    #correct_preds = [0]*self.num_attribute_cat
    train_plot_samples = []
    val_plot_samples = []
    for batch_idx, input_batch in enumerate(loop):
      keypoints, attributes = input_batch
      keypoints, attributes = keypoints.to(self.device), attributes.to(self.device)
      # Get a list of predictions corresponding to different attribute categories
      # For each element of the list, we predict an attribute among those that belong in this category
      #prediction_list = self.model(keypoints)
      if len(train_plot_samples) < self.config['Training']['n_samples_visualization']:
        train_plot_samples.append([torch.flatten(keypoints.detach().cpu(), start_dim=1)[0, :].detach().cpu().numpy(), attributes[0].detach().cpu().numpy()])
      pred = self.model(torch.flatten(keypoints, start_dim=1))
      loss = 0
      targets = attributes.detach().cpu().clone().squeeze(1)
      loss += self.criterion(pred, targets.to(self.device))
        
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      
      avg_loss += loss.item()
      intermediate_loss += loss.item()
      if (batch_idx+1)%log_interval == 0:
        if self.config['wandb']['enable']:
          wandb.log({
              "intermediate_loss": intermediate_loss/log_interval
          })
        intermediate_loss = 0
      #break
    avg_loss = avg_loss/len(train_loader.dataset)
    #avg_acc = correct_preds/len(train_loader)
    

    # Evaluate model on the validation set
    avg_val_loss = 0
    correct_preds_val = [0] * self.num_attribute_cat
    list_pr_labels_val = [ [] for _ in range(self.num_attribute_cat)]
    list_pr_out_val = [ [] for _ in range(self.num_attribute_cat)]
    list_pr_out_val_argmax = [ [] for _ in range(self.num_attribute_cat)]
    with torch.no_grad():
      self.model.eval()
      for batch_idx, input_batch in enumerate(tqdm(val_loader)):
        keypoints, attributes = input_batch
        keypoints, attributes = keypoints.to(self.device), attributes.to(self.device)
        if len(val_plot_samples) < self.config['Training']['n_samples_visualization']:
          val_plot_samples.append([torch.flatten(keypoints.detach().cpu(), start_dim=1)[0, :].detach().cpu().numpy(), attributes[0].detach().cpu().numpy()])
        # Get a list of predictions corresponding to different attribute categories
        # For each element of the list, we predict an attribute among those that belong in this category
        #prediction_list = self.model(keypoints)
        prediction_list = [self.model(torch.flatten(keypoints, start_dim=1))]
        for i, pred in enumerate(prediction_list):
          targets = attributes[:, i].detach().cpu().clone()
          avg_val_loss += self.criterion(pred, targets.to(self.device))
          
          #Get argmax of predictions and check if they are correct
          pred_argmax_val = pred.data.max(1, keepdim=True)[1]
          correct_preds_val[i] += pred_argmax_val.eq(targets.to(self.device).view_as(pred_argmax_val)).sum()
          if len(list_pr_out_val[i]) == 0:
            list_pr_out_val[i] = nn.functional.softmax(pred, dim=-1).detach().cpu().clone().numpy()
            list_pr_out_val_argmax[i] = pred_argmax_val.detach().cpu().clone().numpy().flatten()
            list_pr_labels_val[i] = targets.numpy()
          else:
            list_pr_out_val[i] = np.append(list_pr_out_val[i], nn.functional.softmax(pred, dim=-1).detach().cpu().clone().numpy(), axis=0)
            list_pr_labels_val[i] = np.append(list_pr_labels_val[i], targets.numpy())
            list_pr_out_val_argmax[i] = np.append(list_pr_out_val_argmax[i], pred_argmax_val.detach().cpu().clone().flatten().numpy())
    for i in range(self.num_attribute_cat):
      wandb.log({"conf_mats/conf_mat_{}".format(i) : wandb.plot.confusion_matrix(probs=list_pr_out_val[i],
                        y_true=list_pr_labels_val[i], preds=None)})
      wandb.log({"conf_mats/conf_mat_argmax_{}".format(i) : wandb.plot.confusion_matrix(probs=None,
                        y_true=list_pr_labels_val[i], preds=list_pr_out_val_argmax[i])})
      f1_scores = f1_score(list_pr_labels_val[i], list_pr_out_val_argmax[i], average=None)
      conf_matrix = confusion_matrix(list_pr_labels_val[i], list_pr_out_val_argmax[i])
      accuracies = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
      for j in range(len(f1_scores)):
        converted_label = Person.pred_list_to_str([j], communicative=not self.use_merge)[0]
        wandb.log({"f1_scores/f1_score_{}_{}".format(i, converted_label) : f1_scores[j]})
        wandb.log({"accuracies/accuracy_{}_{}".format(i, converted_label) : accuracies[j]})

    self.show_comparison(train_plot_samples, val_plot_samples)
    #if self.config['wandb']['enable']:prediction_list
    #  self.show_comparison(training_samples_to_plot, validation_samples_to_plot)

    return avg_loss, avg_val_loss/len(val_loader.dataset)

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
        self.scheduler.step(val_loss)

        #Display results
        print(f"Loss epoch {epoch}: ", loss)
        print(f"Validation Loss epoch {epoch}: ", val_loss)

        #Log results in wandb
        if self.config['wandb']['enable']:
          wandb.log({
              "loss": loss, 
              "epoch": epoch,
              "val_loss": val_loss,
          })
        
        #Save best model
        if (self.config['Training']['save_checkpoint'] and val_loss < best_loss):
            best_loss = val_loss
            save_checkpoint(self.model, filename=self.path_model)
            
  def show_comparison(self, training_samples, validation_samples):
    """
      Runs an inference on some samples in the validation set and 
      compares the result between the masked input and the predicted 
      output. 
    """
    print("Generating some samples on the training set")
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=self.config['Training']['n_samples_visualization'], figsize=(10,10))
    fig.suptitle("Example predictions on the training set", fontsize=20)
    kps_painter = KeypointPainter()
    with torch.no_grad():
      for i, training_sample in enumerate(training_samples):
        plt.axis('off')
        x, y, v = convert_xyc_numpy(training_sample[0])
        axes[i].invert_yaxis()
        axes[i].axis('off')
        kps_painter._draw_skeleton(axes[i], x, y, v, skeleton=COCO_PERSON_SKELETON,  mask_joints=False)
      
        pred = self.model(torch.tensor(training_sample[0]).to(self.device).unsqueeze(0))
        pred_argmax = pred.data.max(1, keepdim=True)[1].item()
        target = training_sample[1].item()

        target = Person.pred_list_to_str([target], communicative=not self.use_merge)[0]
        pred_argmax = Person.pred_list_to_str([pred_argmax], communicative=not self.use_merge)[0]

        axes[i].text(0, 0, f"Label: {target}, Pred: {pred_argmax}", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes, size='x-small')
    if self.config['wandb']['enable']:
      plot = wandb.Image(plt)
      wandb.log(
        {
          "training_plot":plot
        }
      )
    
    print("Generating some samples on the validation set")
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=self.config['Training']['n_samples_visualization'], figsize=(10,10))
    fig.suptitle("Example predictions on the validation set", fontsize=20)
    kps_painter = KeypointPainter()
    with torch.no_grad():
      for i, validation_sample in enumerate(validation_samples):
        plt.axis('off')
        x, y, v = convert_xyc_numpy(validation_sample[0])
        axes[i].invert_yaxis()
        axes[i].axis('off')
        kps_painter._draw_skeleton(axes[i], x, y, v, skeleton=COCO_PERSON_SKELETON,  mask_joints=False)
      
        pred = self.model(torch.tensor(validation_sample[0]).to(self.device).unsqueeze(0))
        pred_argmax = pred.data.max(1, keepdim=True)[1].item()
        target = validation_sample[1].item()
        #plt.text(i*10,y, f"Label: {target}, Pred: {pred_argmax}")

        target = Person.pred_list_to_str([target], communicative=not self.use_merge)[0]
        pred_argmax = Person.pred_list_to_str([pred_argmax], communicative=not self.use_merge)[0]

        axes[i].text(0, 0, f"Label: {target}, Pred: {pred_argmax}", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes, size='x-small')
    #plt.show()
    if self.config['wandb']['enable']:
      plot = wandb.Image(plt)
      wandb.log(
        {
          "validation_plot":plot
        }
      )