import torch
import numpy as np

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


def convert_keypoints(keypoints_array, normalize=False, im_size=None, body_parts=False):
    """
        Convert the output from Pifpaf to an interpretable keypoints format
    """
    X, Y, C = np.array(keypoints_array[0]), np.array(keypoints_array[1]), np.array(keypoints_array[2])
    if normalize:
        assert im_size
        X = X / max(im_size)
        Y = Y / max(im_size)
    output_processed_kps = []
    i = 0
    for i in range(len(X)):
        output_processed_kps.append(X[i])
        output_processed_kps.append(Y[i])
        output_processed_kps.append(C[i])
    if body_parts:
        return torch.stack(torch.split(torch.tensor(output_processed_kps), 3), dim=0)
    return torch.tensor(output_processed_kps)

def convert_keypoints_json_input(keypoints_array, normalize=False, im_size=None, body_parts=False):
    """
        Convert the output from Pifpaf to an interpretable keypoints format
    """
    output_processed_kps = []
    n = 1
    if normalize:
        n = max(im_size)

    for i in range(len(keypoints_array)):
        if (i+1)%3 == 0:
            output_processed_kps.append(keypoints_array[i])
        else:
            output_processed_kps.append(keypoints_array[i]/n)
    if body_parts:
        return torch.stack(torch.split(torch.tensor(output_processed_kps), 3), dim=0)
    return torch.tensor(output_processed_kps)