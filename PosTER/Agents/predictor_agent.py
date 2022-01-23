import os, errno
import torch
import cv2
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix

from PosTER.Agents.utils_agent import get_model_for_fine_tuning, load_checkpoint
from PosTER.Datasets.titan_dataset import Person
from PosTER.Models.PosTER import PosTER
from PosTER.Models.PosTER_FT import PosTER_FT
from PosTER.Datasets.dynamic_dataset_prediction import DynamicDatasetPrediction

class Predictor_Agent(object):
    def __init__(self, config, input_folder, output_folder='output'):
        self.config = config
        n_classes = 5
        poster_model = PosTER(config)
        self.model = PosTER_FT(poster_model, n_classes)
        checkpoint_file = os.path.join(self.config['General']['Model_path'], 'PosTER_FT.p')
        load_checkpoint(checkpoint_file, self.model)
        self.dynamicDataset = DynamicDatasetPrediction(input_folder)
        self.attributes_dictionnary = {i:Person.pred_list_to_str([i]) for i in range(n_classes)}
        self.attributes_colors = {Person.pred_list_to_str([i])[0]:int(255/(n_classes+1))*(i+1) for i in range(n_classes)}
        self.model.eval()
        try:
            os.makedirs(output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.output_folder = output_folder
    def predict(self):
        with torch.no_grad():
            for im_name, original_im, kps, boxes in self.dynamicDataset:
                predictions = torch.argmax(self.model(kps), dim=-1).cpu().numpy().tolist()
                converted_predictions = [self.attributes_dictionnary[p] for p in predictions]
                converted_im = self.paint_output(original_im, converted_predictions, boxes)
                Image.fromarray(converted_im).save(os.path.join(self.output_folder, im_name))
    def paint_output(self, original_im, predictions, boxes):
        array_im = np.array(original_im)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2, _ = b
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = self.attributes_colors[predictions[i][0]]
            array_im = cv2.rectangle(array_im, (x1, y1), (x2, y2), (0, color, 0), 1, cv2.LINE_4)
            
            height_small_box = int(0.08*abs(y2-y1))
            width_start = int(0.1*abs(x2-x1))
            fontScale = (2.1/23)*height_small_box # hardcoded value
            font = cv2.FONT_HERSHEY_PLAIN
            
            array_im = cv2.rectangle(array_im, (x1, y1), (x2, y1-height_small_box), (0, color, 0), -1)
            cv2.putText(img=array_im, text=str(predictions[i][0]), org=(x1+width_start, y1-2), fontFace=font, fontScale=fontScale, color=(255, 255, 255),thickness=1)
        return array_im