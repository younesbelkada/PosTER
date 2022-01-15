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
        
        self.fc = nn.Linear(self.pretrained_poster.dim_embed, 5)
        self.classifier = nn.Linear(self.pretrained_poster.dim_embed*18, 5)

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.stack(torch.split(x, 3, dim=1), dim=1)
    
        embed = self.pretrained_poster.poster_embedding(x)
        output_embed = self.pretrained_poster.transformer_encoder(embed)
        cls_token = output_embed[:,0,:]
        
        out =  torch.flatten(output_embed, start_dim=1)
        #out = self.fc(cls_token)
        out = self.classifier(out)
        return out

