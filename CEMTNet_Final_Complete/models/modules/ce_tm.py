import torch
import torch.nn as nn

class CETM(nn.Module):
    def __init__(self, hidden_dim):
        super(CETM, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)        # B x T x 2H
        pooled = rnn_out.mean(dim=1)    # B x 2H
        return self.fc(pooled)          # B x H
