import torch
import torch.nn as nn

class EMSA(nn.Module):
    def __init__(self, hidden_dim):
        super(EMSA, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # B x T x T
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, V)  # B x T x H
        return output
