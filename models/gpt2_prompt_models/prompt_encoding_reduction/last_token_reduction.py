from torch import nn, Tensor


class LastTokenReduction(nn.Module):
    def __init__(self):
        super(LastTokenReduction, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x[:, -1, :]

