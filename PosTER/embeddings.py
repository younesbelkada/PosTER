import math
import torch
import torch.nn as nn



class PositionalEmbedding(nn.Module):
    def __init__(self, n_embed, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, n_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embed, 2).float() * (-math.log(10000.0) / n_embed)
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
    def __init__(self, vocab_size, n_embed, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = PositionalEmbedding(n_embed=n_embed)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, sequence):
        x = self.token_embed(sequence) + self.position_embed(sequence) 
        return self.dropout(x)