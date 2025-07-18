import torch
import torch.nn as nn
from utility import *


class MLPModel(nn.Module):
    def __init__(self, feature_size=model_feature_size, hidden_size=512, dropout=0.4):
        super(MLPModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # 输出维度为1
        )

    def forward(self, src):
        # src: [batch_size, seq_len, feature_size]
        out = self.model(src)  # 输出: [batch_size, seq_len, 1]
        return out.squeeze(-1)  # squeeze掉最后一维，返回 [batch_size, seq_len]
