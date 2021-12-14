import random
import torch
import numpy as np

from PosTER.utils import convert_xyc, convert_keypoints, convert_keypoints_batch

class NormalizeKeypoints(object):
    """
    Normalize the input keypoints.
    Args:
        im_size (tuple): Size of the image
    """
    def __init__(self, im_size):
        assert isinstance((im_size), (tuple))
        self.n = max(im_size)
    def __call__(self, keypoints):
        keypoints_xyc = convert_xyc(keypoints)
        normalized_X = np.array(keypoints_xyc[0])/self.n
        normalized_Y = np.array(keypoints_xyc[1])/self.n
        C = np.array(keypoints_xyc[2])
        return convert_keypoints_batch([normalized_X, normalized_Y, C])
        
class RandomTranslation(object):
    """
    Randomly translate the input keypoints with a given distance of distance.
    Args:
        p (float): Probability when applying the transformation
        distance (int): distance for translation
        max_size (tuple): maximum size of the image to avoid coordinates overflow
    """

    def __init__(self, p, distance, max_size):
        assert isinstance((p, distance, max_size), (float, int, tuple))
        self.p = p
        self.distance = distance
        self.max_width, self.max_height = max_size

    def __call__(self, keypoints):
        if random.uniform(0, 1) <= self.p:
            random_distance_x = random.uniform(-self.distance, self.distance)
            random_distance_y = random.uniform(-self.distance, self.distance)
            X, Y, C = convert_xyc(keypoints)
            X, Y, C = np.array(X), np.array(Y), np.array(C)
            max_x, max_y = np.max(X), np.max(Y)
            min_x, min_y = np.min(X), np.min(Y)
            while (random_distance_x + max_x > self.max_width) or (random_distance_x + min_x < 0) or (random_distance_y + max_y > self.max_height) or (random_distance_y + min_y < 0) : 
                random_distance_x = random.uniform(-self.distance, self.distance)
                random_distance_y = random.uniform(-self.distance, self.distance)
            normalized_X = (X+random_distance_x)
            normalized_Y = (Y+random_distance_y)
            keypoints = convert_keypoints_batch([normalized_X, normalized_Y, C])
        return keypoints

class BodyParts(object):
    """
    Cut the input keypoints into 17 body parts.
    """
    def __init__(self):
        pass 
    def __call__(self, keypoints):
        return torch.stack(torch.split(keypoints, 3, dim=1), dim=1)

