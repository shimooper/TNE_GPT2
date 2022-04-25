from torch import nn, Tensor


class StopPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        """
        input_dim - the dim of the input vector
        hidden_dim - the size of the MLP's hidden dimension
        num_layers - the number of MLP layers
        """
        super(StopPredictor, self).__init__()

        assert (num_layers >= 0)
        assert (hidden_dim >= 0)
        assert (input_dim >= 0)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * num_layers,
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Projecting the input vectors x to 2 class prediction vector
        """
        x = self.projection(x)
        y_hat = self.classifier(x)
        return y_hat
