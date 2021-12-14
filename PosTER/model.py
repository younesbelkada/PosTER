import torch
import torch.nn as nn
from embeddings import PosTEREmbedding

class TransformerModel(nn.Module):
    def __init__(self, dim_token, dim_embed, nhead, d_model, dim_ff, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        ntokens: the size of vocabulary
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        self.poster_embedding = PosTEREmbedding(dim_token, dim_embed, dropout=dropout) 
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) 
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.poster_embedding(src)
        output = self.transformer_encoder(src, src_mask)
        return output


    
