import torch
import torch.nn as nn
from PosTER.embeddings import PosTEREmbedding

class PosTER(nn.Module):
    def __init__(self, config):
        super(PosTER, self).__init__()
        '''

        dim_tokens: Input token dimension
        dim_embed: Token embeddings dimension/also output dimension of each encoder block
        dim_ff: Number of units in intermediary pointwise_ff layer of encoder block
        nlayers: Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: Number of attention heads
        dropout: Dropout probability
        '''
        self.config = config['Model']['PosTER']
        dim_tokens = self.config['dim_tokens'] 
        dim_embed = self.config['dim_embed']
        dim_ff = self.config['dim_ff']
        nlayers = self.config['nlayers']  
        nhead = self.config['nhead']  
        dropout = self.config['dropout']

        self.poster_embedding = PosTEREmbedding(dim_tokens, dim_embed, dropout=dropout) 
        encoder_layers = nn.TransformerEncoderLayer(dim_embed, nhead, dim_ff, dropout) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) 
        self.token_prediction_layer = nn.Linear(dim_embed, dim_tokens)
        self.regressionhead = RegressionHead()

    def forward(self, src):
        embed = self.poster_embedding(src)
        output_embed = self.transformer_encoder(embed)
        cls_token = output_embed[:,0,:]
        output_token = self.token_prediction_layer(output_embed[:, 1:, :])
        x  = self.regressionhead(output_token)
        return  cls_token, torch.flatten(x, start_dim=1)
    
    def load(self, device):
        self.load_state_dict(torch.load())


class PosTER_FT(nn.Module):
    def __init__(self, pretrained_poster, prediction_heads):
        """
            PosTER architecture for fine-tuning
            :- pretrained_poster -: PosTER model
            :- prediction_heads -: nn.ModuleList -> list of prediction heads
        """
        super(PosTER_FT, self).__init__()
        self.pretrained_poster = pretrained_poster
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
        self.prediction_heads = prediction_heads

    def forward(self, x):
        cls_token, _ = self.pretrained_poster(x)
        cls_token = self.fc(cls_token)
        return self.prediction_heads(cls_token)

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