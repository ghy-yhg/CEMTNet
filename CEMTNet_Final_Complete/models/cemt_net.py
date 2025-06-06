import torch
import torch.nn as nn
from models.modules.ccim import CCIM
from models.modules.emsa import EMSA
from models.modules.ce_tm import CETM

class CEMTNet(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, hidden_dim=256, num_classes=2):
        super(CEMTNet, self).__init__()
        self.ccim = CCIM(text_dim, audio_dim, hidden_dim)
        self.emsa = EMSA(hidden_dim)
        self.cetm = CETM(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_feat, audio_feat):
        fused = self.ccim(text_feat, audio_feat)              # B x T x H
        attended = self.emsa(fused)                            # B x T x H
        temporal = self.cetm(attended)                         # B x H
        logits = self.classifier(temporal)                     # B x C
        return logits
