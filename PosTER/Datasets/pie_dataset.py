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

from PosTER.Datasets.predictor import PifPafPredictor
from PosTER.Datasets.transforms_agent import TransformsAgent
from PosTER.Datasets.utils import preprocess_pifpaf, prepare_pif_kps, convert_keypoints, convert_keypoints_json_input, get_input_path
from PosTER.Datasets.augmentations import NormalizeKeypoints, BodyParts, RandomTranslation, RandomMask, NormalizeKeypointsRelative

def my_collate(batch):
    # TO DO: Filter out empty arrays
    #batch = filter (lambda x:len(x) != 0, batch)
    #for b in batch:
    #    print(b.shape)
    batch = torch.cat(batch, dim=-1)
    return batch.permute(2, 0, 1)

class DynamicDataset(Dataset):
    """
        Class definition for the dynamic keypoints dataset.
        Expected inputs: path to images
        Expected output: unlabeled human keypoints
    """
    def __init__(self, config, split=None):
        self.config = config["Dataset"]
        #if split:
        #    self.path_images = get_input_path(self.config["DynamicDataset"]['path_images'], self.config["DynamicDataset"]['ext'], split)
        #else:
        #    self.path_images = glob(os.path.join(self.config["DynamicDataset"]['path_images'], '**', '*'+self.config["DynamicDataset"]['ext']), recursive=True)
        self.path_images = get_input_path(self.config["DynamicDataset"]['path_input'], self.config["DynamicDataset"]['ext'], split, self.config['split'][split])
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
    def __init__(self, config, split, transforms=None):
        self.config = config["Dataset"]
        #if split:
        #    self.path_images = get_input_path(self.config["DynamicDataset"]['path_images'], self.config["DynamicDataset"]['ext'], split)
        #else:
        #    self.path_images = glob(os.path.join(self.config["DynamicDataset"]['path_images'], '**', '*'+self.config["DynamicDataset"]['ext']), recursive=True)
        self.path_kps = get_input_path(self.config["StaticDataset"]['path_input'], self.config["StaticDataset"]['ext'], split, self.config['split'][split])
        self.im_size = (self.config["StaticDataset"]["im_width"], self.config["StaticDataset"]["im_height"])
        self.transforms = transforms

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
            if self.transforms:
                predicted_kps = self.transforms(predicted_kps)
        else:
            predicted_kps = torch.tensor([])
        return predicted_kps.float()
