import torch
import torch.nn as nn
import math
from utility import *


class LSTMModel(nn.Module):

    def __init__(self, feature_size=model_feature_size, hidden_size=model_d_model, num_layers=model_num_layers,
                 dropout=model_dropout, device='cuda'):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Linear(feature_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, src):
        # src: [batch_size, seq_len, feature_size]
        # 输入embedding
        src = self.embedding(src)  # [batch_size, seq_len, hidden_size]

        # LSTM层
        lstm_out, _ = self.lstm(src)  # lstm_out: [batch_size, seq_len, hidden_size]

        # 输出层
        output = self.linear(lstm_out)  # [batch_size, seq_len, out_size]
        output_squeeze = output.squeeze()
        return output_squeeze


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
