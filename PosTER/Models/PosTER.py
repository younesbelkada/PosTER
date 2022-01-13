import torch
import torch.nn as nn
from PosTER.Models.embeddings import PosTEREmbedding
from PosTER.Models.utils_models import RegressionHead, PredictionHeads

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
        self.dim_embed = self.config['dim_embed']
        dim_ff = self.config['dim_ff']
        nlayers = self.config['nlayers']  
        nhead = self.config['nhead']  
        dropout = self.config['dropout']
        self.enable_bt = config["Training"]['criterion']['enable_bt']
        
        self.poster_embedding = PosTEREmbedding(dim_tokens, self.dim_embed, dropout=dropout) 
        encoder_layers = nn.TransformerEncoderLayer(self.dim_embed, nhead, dim_ff, dropout) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) 
        self.token_prediction_layer = nn.Linear(self.dim_embed, dim_tokens)
        self.regressionhead = RegressionHead()
        self.bt_head = self.get_head()

    def forward(self, src):
        embed = self.poster_embedding(src)
        output_embed = self.transformer_encoder(embed)
        cls_token = output_embed[:,0,:]
        output_token = self.token_prediction_layer(output_embed[:, 1:, :])
        x  = self.regressionhead(output_token)
        
        if self.enable_bt:
            return  self.bt_head(cls_token), torch.flatten(x, start_dim=1)
        else:
            return  cls_token, torch.flatten(x, start_dim=1)
    
    def load(self, device):
        self.load_state_dict(torch.load())
        
    def get_head(self):
        # first layer
        proj_layers = [nn.Linear(self.dim_embed, 512, bias=False)]
        for i in range(2):
            proj_layers.append(nn.BatchNorm1d(512))
            proj_layers.append(nn.ReLU(inplace=True))
            proj_layers.append(nn.Linear(512, 512, bias=False))
            
        return nn.Sequential(*proj_layers)