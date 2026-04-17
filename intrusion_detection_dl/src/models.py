"""DNN, LSTM, and hybrid DNN+LSTM classifiers."""

from __future__ import annotations

import torch
from torch import nn


class DNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64, 32), num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class HybridDNNLSTM(nn.Module):
    """Per-sequence timestep passed through a small MLP; LSTM over timestep embeddings."""

    def __init__(self, n_features: int, mlp_hidden: int = 32, lstm_hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=mlp_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        z = self.mlp(x.reshape(b * t, f)).reshape(b, t, -1)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :])
