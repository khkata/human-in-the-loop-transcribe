import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_prob=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_prob)
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)