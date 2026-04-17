
from __future__ import annotations

import torch
from torch import nn

DNN_HIDDEN = (128, 64, 32)
DROPOUT_P = 0.2
LSTM_HIDDEN = 64


def _mlp_backbone(in_dim: int, hidden: tuple[int, ...], dropout_p: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout_p)]
        d = h
    return nn.Sequential(*layers), d


class DNNClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = DNN_HIDDEN,
        dropout_p: float = DROPOUT_P,
        num_classes: int = 2,
    ):
        super().__init__()
        body, d = _mlp_backbone(in_dim, hidden, dropout_p)
        self.net = nn.Sequential(body, nn.Linear(d, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMClassifier(nn.Module):
   

    def __init__(
        self,
        n_features: int,
        hidden: int = LSTM_HIDDEN,
        num_layers: int = 1,
        num_classes: int = 2,
    ):
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


    def __init__(
        self,
        n_features: int,
        dnn_hidden: tuple[int, ...] = DNN_HIDDEN,
        dropout_p: float = DROPOUT_P,
        lstm_hidden: int = LSTM_HIDDEN,
        num_classes: int = 2,
    ):
        super().__init__()
        self.mlp, lstm_in = _mlp_backbone(n_features, dnn_hidden, dropout_p)
        self.lstm = nn.LSTM(
            input_size=lstm_in,
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
