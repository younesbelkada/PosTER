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


def bt_loss(z1, z2, lmbda):
    
  #Normalize the projector's output across the batch
  norm_z1 = (z1 - z1.mean(0))/ z1.std(0)
  norm_z2 = (z2 - z2.mean(0))/ z2.std(0)

  #Cross correlation matrix
  batch_size = z1.size(0)
  cc_M = torch.einsum('bi,bj->ij', (norm_z1, norm_z2)) / batch_size

  #Invariance loss
  diag = torch.diagonal(cc_M)
  invariance_loss = ((torch.ones_like(diag) - diag) ** 2).sum()

  #Zero out the diag elements and flatten the matrix to compute the loss
  cc_M.fill_diagonal_(0)
  redundancy_loss = (cc_M.flatten() ** 2 ).sum()
  loss = invariance_loss + lmbda * redundancy_loss

  return loss