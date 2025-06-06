import torch
import torch.nn as nn

class CCIM(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim):
        super(CCIM, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.align_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feat, audio_feat):
        t = self.text_proj(text_feat)        # B x T x H
        a = self.audio_proj(audio_feat)      # B x T x H
        concat = torch.cat([t, a], dim=-1)   # B x T x 2H
        gate = self.align_gate(concat)       # B x T x 1
        fused = gate * t + (1 - gate) * a    # B x T x H
        return fused
