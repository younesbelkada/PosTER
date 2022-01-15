import os
import torch
import numpy as np

from glob import glob
import torch.optim as optim

def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]

def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints

def convert_keypoints(keypoints_array):
    """
        Convert the output from PifPaf to an interpretable keypoints format
    """
    X, Y, C = np.array(keypoints_array[0]), np.array(keypoints_array[1]), np.array(keypoints_array[2])
    output_processed_kps = []
    for i in range(len(X)):
        output_processed_kps.append(X[i])
        output_processed_kps.append(Y[i])
        output_processed_kps.append(C[i])
    output_processed_kps = torch.tensor(np.array(output_processed_kps))
    #output_processed_kps[output_processed_kps.clone().detach().requires_grad_(True) < 0] = 0
    return output_processed_kps

def convert_keypoints_batch(keypoints_array):
    """
        Convert the output from PifPaf to an interpretable keypoints format
    """
    X, Y, C = keypoints_array
    output_processed_kps = []
    for i in range(len(X)):
        output_processed_kps.append(X[i, :].tolist())
        output_processed_kps.append(Y[i, :].tolist())
        output_processed_kps.append(C[i, :].tolist())
    output_processed_kps = torch.tensor(np.array(output_processed_kps))
    #output_processed_kps[output_processed_kps.clone().detach().requires_grad_(True) < 0] = 0
    return torch.transpose(output_processed_kps, 0, 1)
    #return output_processed_kps

def convert_keypoints_json_input(keypoints_array):
    """
        Convert the output from Pifpaf to an interpretable keypoints format
    """
    return torch.tensor(keypoints_array)

def convert_xyc(keypoints_tensor):
    """
        Convert the kps tensor into X, Y, C format
    """
    if len(keypoints_tensor.shape) == 1:
        keypoints_tensor = keypoints_tensor.unsqueeze(0)
    X, Y, C = [], [], []
    i = 0
    while i < len(keypoints_tensor[0, :]):
        X.append(keypoints_tensor[:, i].detach().cpu().numpy())
        Y.append(keypoints_tensor[:, i+1].detach().cpu().numpy())
        C.append(keypoints_tensor[:, i+2].detach().cpu().numpy())
        i += 3
    return [X, Y, C]

def convert_xyc_numpy(keypoints):
    """
        Convert the kps tensor into X, Y, C format
        Expects:
            :- keypoints -: np array of size (51,)
    """
    if len(keypoints.shape) == 2:
        keypoints = np.ravel(keypoints)
    X, Y, C = [], [], []
    i = 0
    while i < len(keypoints):
        X.append(keypoints[i])
        Y.append(keypoints[i+1])
        C.append(keypoints[i+2])
        i += 3
    return [np.array(X), np.array(Y), np.array(C)]

def get_input_path(path_input, ext, split=None, array_sets=None):
    if split:
        assert split in ['train', 'val']
        paths = []
        for sets in array_sets:
            paths.extend(glob(os.path.join(path_input, sets, '**', '*'+ext), recursive=True))
    else:
        paths = glob(os.path.join(path_input, '**', '*'+ext), recursive=True)
    return paths

def from_stats_to_weights(stat_dicts):
    weights_dict = {}
    for categories in stat_dicts.keys():
        weights_dict[categories] = 1. - (np.array(stat_dicts[categories]) / sum(stat_dicts.values()) )
    return weights_dict

def return_weights(weights_dict, labels):
    final_weights = np.zeros(labels.shape[0])
    for i, array_labels in enumerate(labels):
        for j in range(len(array_labels)):
            final_weights[i] += weights_dict[j][array_labels[j]]
            break
    return 1-final_weights