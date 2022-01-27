import wandb
import os
import torch
import torch.nn as nn
import numpy as np 

import matplotlib as mpl
mpl.use('TkAgg')

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from PosTER.Agents.utils_agent import get_model_for_fine_tuning, load_checkpoint, get_model_for_testing
from PosTER.Datasets.titan_dataset import Person

class Evaluator(object):
    """
        Wrapper class for the evaluator
    """
    def __init__(self, config):
        self.config = config
        self.use_merge = config['Dataset']['TITAN']['use_merge']
        if config['General']['DatasetType'] == 'TITAN':
            if self.use_merge:
                n_classes = 5
            else:
                n_classes = 4
            self.num_attribute_cat = 1
        else:
            raise "Not implemented"
        self.model = get_model_for_testing(config, n_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def evaluate(self, test_dataloader):
        if self.config['wandb']['enable']:
            wandb_entity = self.config['wandb']['entity']
            wandb.init(project=self.config['wandb']['project_name'], entity=wandb_entity, config=self.config)

        correct_preds_val = 0
        list_pr_labels_val = []
        list_pr_out_val = []
        list_pr_out_val_argmax = []
        with torch.no_grad():
            self.model.eval()
            for batch_idx, input_batch in enumerate(tqdm(test_dataloader)):
                keypoints, attributes = input_batch
                keypoints, attributes = keypoints.to(self.device), attributes.to(self.device)
                
                # Get a list of predictions corresponding to different attribute categories
                # For each element of the list, we predict an attribute among those that belong in this category
                #prediction_list = self.model(keypoints)
                prediction = self.model(torch.flatten(keypoints, start_dim=1))
                
                targets = attributes[:, 0].detach().cpu().clone()
                    
                #Get argmax of predictions and check if they are correct
                pred_argmax_val = prediction.data.max(1, keepdim=True)[1]

                correct_preds_val += pred_argmax_val.eq(targets.to(self.device).view_as(pred_argmax_val)).sum()
                if len(list_pr_out_val) == 0:
                    list_pr_out_val = nn.functional.softmax(prediction, dim=-1).detach().cpu().clone().numpy()
                    list_pr_out_val_argmax = pred_argmax_val.detach().cpu().clone().flatten().numpy()
                    list_pr_labels_val = targets.numpy()
                else:
                    list_pr_out_val = np.append(list_pr_out_val, nn.functional.softmax(prediction, dim=-1).detach().cpu().clone().numpy(), axis=0)
                    list_pr_labels_val = np.append(list_pr_labels_val, targets.numpy())
                    list_pr_out_val_argmax = np.append(list_pr_out_val_argmax, pred_argmax_val.detach().cpu().clone().flatten().numpy())
        wandb.log({"conf_mat_test_set" : wandb.plot.confusion_matrix(probs=list_pr_out_val,
                y_true=list_pr_labels_val, preds=None)})
        wandb.log({"conf_mat_argmax_test_set" : wandb.plot.confusion_matrix(probs=None,
                y_true=list_pr_labels_val, preds=list_pr_out_val_argmax)})
        f1_scores = f1_score(list_pr_labels_val, list_pr_out_val_argmax, average=None)
        conf_matrix = confusion_matrix(list_pr_labels_val, list_pr_out_val_argmax)
        accuracies = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
        for j in range(len(f1_scores)):
            converted_label = Person.pred_list_to_str([j], communicative=not self.use_merge)[0]
            wandb.log({"f1_scores/f1_score_test_{}".format(converted_label) : f1_scores[j]})
            wandb.log({"accuracies/accuracy_test_{}".format(converted_label) : accuracies[j]})