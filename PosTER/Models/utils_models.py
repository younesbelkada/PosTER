import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(
            51, 51
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.sigmoid(self.fc(x))

class PredictionHeads(nn.Module):
    def __init__(self, list_attributes):
        """
            Prediction Heads for PosTER fine tuning
            :- list_attributes -: list containing number of attributes per category
        """
        super(PredictionHeads, self).__init__()
        self.nb_heads = len(list_attributes)
        assert self.nb_heads > 0
        
        self.list_modules = []
        for e in list_attributes:
            self.list_modules.append(nn.Linear(1024, e))
        self.list_modules = nn.ModuleList(self.list_modules)
    
    def forward(self, x):
        return [head(x) for head in self.list_modules]

class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 1028),
            nn.ReLU(True),
            nn.BatchNorm1d(1028),
            nn.Linear(1028, 1028),
            nn.ReLU(True),
            nn.Linear(1028, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(True),
            nn.Linear(1028, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(True),
            nn.Linear(1028, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(True),
            nn.Linear(1028, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(True),
            nn.Linear(1028, 5)
        )
    def forward(self, x):
        return self.net(x)