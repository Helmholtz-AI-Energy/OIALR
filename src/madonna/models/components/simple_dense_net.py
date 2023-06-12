from __future__ import annotations

from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        fc_sizes: list[int] | None = None,
        output_size: int = 10,
        activation: str = "ReLU",
        batch_norm: bool = True,
        bias: bool = True,
        activate_last_layer: bool = True,
    ):
        super().__init__()
        if fc_sizes is None:
            fc_sizes = [10, 10, 10]
        self.activation = getattr(nn, activation)()

        layers = []
        last_sz = input_size
        for sz in fc_sizes:
            layers.append(nn.Linear(last_sz, sz, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sz))
            layers.append(self.activation)
            last_sz = sz

        layers.append(nn.Linear(fc_sizes[-1], output_size, bias=bias))
        if activate_last_layer:
            layers.append(self.activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
