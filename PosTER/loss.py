import torch
import torch.nn as nn

def point_loss(tensor1, tensor2, eps=1e-15):
    """
        Implementation of the Point loss
        :- tensor1 -: First input tensor
        :- tensor2 -: Second input tensor
        :- eps -: Regularization term
    """
    stacked_tensor = torch.stack((tensor1, tensor2), dim=0)
    min_tensors = torch.min(stacked_tensor, dim=0)[0] + eps
    max_tensors = torch.max(stacked_tensor, dim=0)[0] + eps
    return -torch.log(min_tensors/max_tensors).mean()


