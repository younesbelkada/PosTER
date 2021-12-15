"""
def convert_keypoints(keypoints_array, normalize=False, im_size=None, body_parts=False):
    """
#Convert the output from Pifpaf to an interpretable keypoints format
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
    output_processed_kps = torch.tensor(output_processed_kps)
    output_processed_kps[torch.tensor(output_processed_kps) < 0] = 0
    if body_parts:
        return torch.stack(torch.split(output_processed_kps, 3), dim=0)
    return output_processed_kps
"""

"""
def convert_keypoints_json_input(keypoints_array, normalize=False, im_size=None, body_parts=False):
    """
#Convert the output from Pifpaf to an interpretable keypoints format
"""
    output_processed_kps = []
    n = 1
    if normalize:
        n = max(im_size)

    for i in range(len(keypoints_array)):
        if (i+1)%3 == 0 or not normalize:
            output_processed_kps.append(keypoints_array[i])
        else:
            output_processed_kps.append(keypoints_array[i]/n)
    if body_parts:
        return torch.stack(torch.split(torch.tensor(output_processed_kps), 3), dim=0)
    return torch.tensor(output_processed_kps)
"""