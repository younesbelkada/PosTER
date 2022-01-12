import torch
import torch.nn as nn

class PosTER_FT(nn.Module):
    def __init__(self, pretrained_poster, prediction_heads):
        """
            PosTER architecture for fine-tuning
            :- pretrained_poster -: PosTER model
            :- prediction_heads -: nn.ModuleList -> list of prediction heads
        """
        super(PosTER_FT, self).__init__()
        self.pretrained_poster = pretrained_poster
        #self.pretrained_poster.requires_grad_(False)
        '''
        self.fc = nn.Sequential(
            nn.Linear(pretrained_poster.token_prediction_layer.in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        '''
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
      if len(x.shape) == 2:
        x = torch.split(x, 3, dim=1)
      cls_token, _ = self.pretrained_poster(x)
      out = self.fc(cls_token)
      return out

