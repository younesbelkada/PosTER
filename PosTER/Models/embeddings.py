import math
import torch
import torch.nn as nn

from einops import repeat


class PositionalEmbedding(nn.Module):
    '''
    Positional embeddings constructed with sin and cos functions
    '''
    def __init__(self, dim_embed, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_embed, 2).float() * (-math.log(10000.0) / dim_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]

        return x


class PosTEREmbedding(nn.Module):
    """
    PosTER embedding has 2 components:
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
    
    """
    def __init__(self, dim_token, dim_embed, dropout=0.1):
        """
        :param dim_token: dimension of the initial tokens
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_embed = nn.Linear(dim_token, dim_embed)
        self.position_embed = PositionalEmbedding(dim_embed = dim_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(dim_embed)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embed))

    def forward(self, sequence):
        x = self.token_embed(sequence) 
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_embed(x) 
        x = self.layernorm(x)

        return self.dropout(x)