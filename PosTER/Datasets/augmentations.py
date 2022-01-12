import random
import torch
import numpy as np

from PosTER.Datasets.utils import convert_xyc, convert_keypoints, convert_keypoints_batch, convert_xyc_numpy

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
        if torch.is_tensor(keypoints):
            keypoints_xyc = convert_xyc(keypoints.unsqueeze(0))
        else:
            keypoints_xyc = convert_xyc_numpy(keypoints)
        normalized_X = np.array(keypoints_xyc[0])/self.n
        normalized_Y = np.array(keypoints_xyc[1])/self.n
        C = np.array(keypoints_xyc[2])
        if len(normalized_X.shape) == 2:
            return convert_keypoints_batch([normalized_X, normalized_Y, C])
        else:
            return convert_keypoints([normalized_X, normalized_Y, C]).unsqueeze(0)

class NormalizeKeypointsRelative(object):
    """
    convert the key points from absolute coordinate to center+relative coordinate
            all_poses shape (n_samples, n_keypoints, n_dim)
            
        COCO_KEYPOINTS = [
            'nose',            # 0
            'left_eye',        # 1
            'right_eye',       # 2
            'left_ear',        # 3
            'right_ear',       # 4
            'left_shoulder',   # 5
            'right_shoulder',  # 6
            'left_elbow',      # 7
            'right_elbow',     # 8
            'left_wrist',      # 9
            'right_wrist',     # 10
            'left_hip',        # 11
            'right_hip',       # 12
            'left_knee',       # 13
            'right_knee',      # 14
            'left_ankle',      # 15
            'right_ankle',     # 16
        ]

        Args:
            all_poses (np.ndarray): pose array, size (batch_size, V, C)

        Returns:
            converted_poses: size (batch_size, V+1, C)
        """
    def __init__(self):
        pass
    def __call__(self, keypoints):
        if torch.is_tensor(keypoints):
            keypoints_x, keypoints_y, keypoints_c = convert_xyc(keypoints.unsqueeze(0))
        else:
            keypoints_x, keypoints_y, keypoints_c = convert_xyc_numpy(keypoints)
        keypoints_x, keypoints_y, keypoints_c = np.array(keypoints_x), np.array(keypoints_y), np.array(keypoints_c) 
        
        left_shoulder_x = keypoints_x[5]
        right_shoulder_x = keypoints_x[6]
        left_shoulder_y = keypoints_y[5]
        right_shoulder_y = keypoints_y[6]

        left_hip_x = keypoints_x[11]
        right_hip_x = keypoints_x[12]
        left_hip_y = keypoints_y[11]
        right_hip_y = keypoints_y[12]

        top_mid_x = 0.5*(left_shoulder_x + right_shoulder_x)
        bottom_mid_x = 0.5*(left_hip_x + right_hip_x)
        top_mid_y = 0.5*(left_shoulder_y + right_shoulder_y)
        bottom_mid_y = 0.5*(left_hip_y + right_hip_y)

        mid_x = 0.5*(top_mid_x+bottom_mid_x)
        mid_y = 0.5*(top_mid_y+bottom_mid_y)

        relative_coord_x = keypoints_x - mid_x
        relative_coord_y = keypoints_y  - mid_y
        if len(relative_coord_x.shape) == 2:
            return convert_keypoints_batch([relative_coord_x, relative_coord_y, keypoints_c])
        else:
            return convert_keypoints([relative_coord_x, relative_coord_y, keypoints_c]).unsqueeze(0)

        
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
        """
            Tries to translate the keypoints in a random x, y direction if it fits the image size.
            Keeps the original keypoints otherwise
        """
        if random.uniform(0, 1) <= self.p:
            random_distance_x = random.uniform(-self.distance, self.distance)
            random_distance_y = random.uniform(-self.distance, self.distance)
            if torch.is_tensor(keypoints):
                X, Y, C = convert_xyc(keypoints.unsqueeze(0))
            else:
                X, Y, C = convert_xyc_numpy(keypoints)
            X, Y, C = np.array(X), np.array(Y), np.array(C)
            max_x, max_y = np.max(X), np.max(Y)
            min_x, min_y = np.min(X), np.min(Y)
            random_distance_x, random_distance_y = random.uniform(-self.distance, self.distance), random.uniform(-self.distance, self.distance)
            if (random_distance_x + max_x <= self.max_width) and (random_distance_x + min_x >= 0) and (random_distance_y + max_y <= self.max_height) and (random_distance_y + min_y >= 0): 
                random_distance_x = random.uniform(-self.distance, self.distance)
                random_distance_y = random.uniform(-self.distance, self.distance)
                normalized_X = (X+random_distance_x)
                normalized_Y = (Y+random_distance_y)
                if len(normalized_X.shape) == 1:
                    keypoints = convert_keypoints([normalized_X, normalized_Y, C])
                else:
                    keypoints = convert_keypoints_batch([normalized_X, normalized_Y, C])
        return keypoints

class BodyParts(object):
    """
    Cut the input keypoints into 17 body parts.
    """
    def __init__(self):
        pass 
    def __call__(self, keypoints):
        return torch.stack(torch.split(keypoints, 3, dim=1), dim=1).squeeze(0)

class RandomMask(object):
    """
    Randomly mask the N tokens from the input. The method is based on 
    Bernoulli distribution to sample which body part to mask.
    :- N -: Number of body parts to mask (at most)
    Credits to clement.apavou and arthur.zucker for their help
    """
    def __init__(self, N):
        self.N = N 
    def __call__(self, keypoints):
        full_keypoints = keypoints.clone()
        nb_body_parts = keypoints.shape[1]
        
        nb_masks = torch.randint(1, self.N, (1,)).item()

        index_to_mask = torch.ones((keypoints.shape[0], nb_body_parts))*(1-(nb_masks/nb_body_parts))
        masked_keypoints = keypoints*(torch.bernoulli(index_to_mask).unsqueeze(-1))

        nb_masks = torch.randint(1, self.N, (1,)).item()
        index_to_mask = torch.ones((keypoints.shape[0], nb_body_parts))*(1-(nb_masks/nb_body_parts))
        masked_keypoints_for_bt = keypoints*(torch.bernoulli(index_to_mask).unsqueeze(-1))
        return masked_keypoints_for_bt, masked_keypoints, full_keypoints
