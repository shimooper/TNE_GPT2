from torch import nn, Tensor


class SpanPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        """
        input_dim - the dim of the input vector
        hidden_dim - the size of the MLP's hidden dimension
        num_layers - the number of MLP layers
        """
        super(SpanPredictor, self).__init__()

        assert (num_layers >= 0)
        assert (hidden_dim >= 0)
        assert (input_dim >= 0)

        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for i in range(num_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
        self.projection = nn.Sequential(*modules)

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Projecting the input vectors x to 2 coordinate prediction vector
        """
        x = self.projection(x)
        y_hat = self.classifier(x)
        return y_hat
