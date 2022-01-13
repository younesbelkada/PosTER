from tqdm import tqdm
from glob import glob

from torchvision import transforms

from PosTER.Datasets.augmentations import NormalizeKeypoints, BodyParts, RandomTranslation, RandomMask, NormalizeKeypointsRelative, RandomFlip

class TransformsAgent(object):
    """
        Object that manages the transformations to use given the config file
    """
    def __init__(self, config, test=False):
        self.config = config['Dataset']['Transforms']
        self.test = test
    def get_transforms(self, im_size):
        transform = []
        if self.config['translation']['enable']:
            transform.append(RandomTranslation(self.config['translation']['p'] if not self.test else 0, self.config['translation']['distance'], im_size))
        if self.config['normalize']:
            #transform.append(NormalizeKeypoints(im_size))
            transform.append(NormalizeKeypointsRelative())
        if self.config["randomflip"]['enable']:
            transform.append(RandomFlip(self.config['randomflip']['p'] if not self.test else 0 ))
        if self.config['body_parts']:
            transform.append(BodyParts())
        
        if len(transform) > 0:
            return transforms.Compose(transform)
        return None