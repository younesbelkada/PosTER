import torch
import torch.nn as nn

class PosTER_FT(nn.Module):
    def __init__(self, pretrained_poster, n_classes=5):
        """
            PosTER architecture for fine-tuning
            :- pretrained_poster -: PosTER model
            :- prediction_heads -: nn.ModuleList -> list of prediction heads
        """
        super(PosTER_FT, self).__init__()
        self.pretrained_poster = pretrained_poster
        self.pretrained_poster.requires_grad_(False)
        
        #self.classifier = nn.Linear(self.pretrained_poster.dim_embed*18, n_classes)
        self.seq = nn.Sequential(
            nn.Linear(self.pretrained_poster.dim_embed*18, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.stack(torch.split(x, 3, dim=1), dim=1)
    
        embed = self.pretrained_poster.poster_embedding(x)
        output_embed = self.pretrained_poster.transformer_encoder(embed)
        #cls_token = output_embed[:,0,:]
        
        out =  torch.flatten(output_embed, start_dim=1)
        #out = self.fc(cls_token)
        #out = self.classifier(out)
        out = self.seq(out)
        return out


