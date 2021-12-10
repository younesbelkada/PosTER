import os
import openpifpaf
import torch
import PIL

from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PosTER.predictor import PifPafPredictor

class DynamicDataset(Dataset):
    """
        Class definition for the dynamic keypoints dataset.
        Expected inputs: path to images
        Expected output: unlabeled human keypoints
    """
    def __init__(self, config):
        self.config = config["Dataset"]["DynamicDataset"]
        self.path_images = glob(os.path.join(self.config['path_images'], '**', '*'+self.config['ext']))
        self.predictor = PifPafPredictor()
    
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
        predicted_loader = self.predictor.images(image_name_to_predict)
        for i, (pred_batch, _, meta_batch) in enumerate(tqdm(predicted_loader)):
            im_size = PIL.Image.open(open(meta_batch['file_name'], 'rb')).convert('RGB').size
            pifpaf_outs = {
                'json_data': [ann.json_data() for ann in pred_batch]
            }
            im_name = os.path.basename(meta_batch['file_name'])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)

            print(keypoints)


