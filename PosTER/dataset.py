import os
import openpifpaf
import torch
import PIL
import json

from tqdm import tqdm
from glob import glob

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PosTER.predictor import PifPafPredictor
from PosTER.utils import preprocess_pifpaf, prepare_pif_kps, convert_keypoints, convert_keypoints_json_input
from PosTER.augmentations import NormalizeKeypoints, BodyParts, RandomTranslation

def my_collate(batch):
    # TO DO: Filter out empty arrays
    #batch = filter (lambda x:len(x) != 0, batch)
    #print(batch)
    batch = torch.cat(batch, dim=0)
    return batch

class DynamicDataset(Dataset):
    """
        Class definition for the dynamic keypoints dataset.
        Expected inputs: path to images
        Expected output: unlabeled human keypoints
    """
    def __init__(self, config):
        self.config = config["Dataset"]
        self.path_images = glob(os.path.join(self.config["DynamicDataset"]['path_images'], '**', '*'+self.config["DynamicDataset"]['ext']), recursive=True)
        self.predictor_obj = PifPafPredictor()
        self.transforms_agent = TransformsAgent(config)
    
    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.path_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the predicted keypoints from an example image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name_to_predict = [self.path_images[idx]]
        predicted_loader = self.predictor_obj.predictor.images(image_name_to_predict)

        predicted_kps = []

        for i, (pred_batch, _, meta_batch) in enumerate(tqdm(predicted_loader)):
            im_size = PIL.Image.open(open(meta_batch['file_name'], 'rb')).convert('RGB').size
            pifpaf_outs = {
                'json_data': [ann.json_data() for ann in pred_batch]
            }
            im_name = os.path.basename(meta_batch['file_name'])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)

            if len(keypoints) > 0:
                for j in range(len(boxes)):
                    predicted_kps.append(convert_keypoints(keypoints[j]))
            predicted_kps = torch.stack(predicted_kps, dim=0)

            transforms_to_apply = self.transforms_agent.get_transforms(im_size)
            if transforms_to_apply:
                predicted_kps = transforms_to_apply(predicted_kps)
        return predicted_kps

class StaticDataset(Dataset):
    """
        Class definition for the static keypoints dataset.
        Expected inputs: path to pre-computed keypoints
        Expected output: unlabeled human keypoints
    """
    def __init__(self, config):
        self.config = config["Dataset"]
        self.path_kps = glob(os.path.join(self.config["StaticDataset"]['path_joints'], '**', '*'+self.config["StaticDataset"]['ext']), recursive=True)
        self.im_size = (self.config["StaticDataset"]["im_width"], self.config["StaticDataset"]["im_height"])
        self.transforms_agent = TransformsAgent(config)

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.path_kps)
    
    def __getitem__(self, idx):
        """
            Getter function in order to get the predicted keypoints from an example image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        predicted_kps = []
        json_file = json.load(open(self.path_kps[idx], 'r'))
        if len(json_file) > 0:
            for i in range(len(json_file)):
                kps_array = json_file[i]['keypoints']
                predicted_kps.append(convert_keypoints_json_input(kps_array))
            predicted_kps = torch.stack(predicted_kps, dim=0)
            transforms_to_apply = self.transforms_agent.get_transforms(self.im_size)
            if transforms_to_apply:
                predicted_kps = transforms_to_apply(predicted_kps)
        else:
            predicted_kps = torch.tensor([])
        return predicted_kps

class TransformsAgent(object):
    """
        Object that manages the transformations to use given the config file
    """
    def __init__(self, config):
        self.config = config['Dataset']['Transforms']
    def get_transforms(self, im_size):
        transform = []
        if self.config['translation']['enable']:
            transform.append(RandomTranslation(self.config['translation']['p'], self.config['translation']['distance'], im_size))
        if self.config['normalize']:
            transform.append(NormalizeKeypoints(im_size))
        if self.config['body_parts']:
            transform.append(BodyParts())
        
        if len(transform) > 0:
            return transforms.Compose(transform)
        return None