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
        output_token = self.token_prediction_layer(output_embed)
        return self.regressionhead(output_token)

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