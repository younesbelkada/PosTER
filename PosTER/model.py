import torch
import torch.nn as nn
from embeddings import PosTEREmbedding

class PosTER(nn.Module):
    def __init__(self, dim_tokens, dim_embed, dim_ff, nhead, nlayers, dropout=0.5):
        super(PosTER, self).__init__()
        '''

        dim_tokens: Input token dimension
        dim_embed: Token embeddings dimension/also output dimension of each encoder block
        dim_ff: Number of units in intermediary pointwise_ff layer of encoder block
        nlayers: Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: Number of attention heads
        dropout: Dropout probability
        '''

        self.poster_embedding = PosTEREmbedding(dim_tokens, dim_embed, dropout=dropout) 
        encoder_layers = nn.TransformerEncoderLayer(dim_embed, nhead, dim_ff, dropout) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) 
        self.token_prediction_layer = nn.Linear(dim_embed, dim_tokens)

    def forward(self, src):
        embed = self.poster_embedding(src)
        output_embed = self.transformer_encoder(embed)
        output_token = self.token_prediction_layer(output_embed)

        return output_token


if __name__ == "__main__":

    dim_tokens = 3 # the size of vocabulary
    dim_embed = 100  # hidden dimension
    dim_ff = 2048
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 5  # the number of heads in the multiheadattention models
    dropout = 0.1  # the dropout value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = PosTER(dim_tokens, dim_embed, dim_ff, nhead, nlayers, dropout=0.5).to(device)
    dummy_input = torch.tensor([[[0.3, 0.6, 0.2], [0.1, 0.9, 0.8]]], dtype= torch.float).to(device)
    out = model.forward(dummy_input)

    assert out.shape == dummy_input.shape
    print(f"input shape: {dummy_input.shape}")
    print(f"output shape: {out.shape}")

    
