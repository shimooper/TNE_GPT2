from torch import nn, Tensor
import torch


class SpanStopPredictor(nn.Module):
    def __init__(self, span_predictor: nn.Module):
        """
        span_predictor - module predicting (start,end) tokens of the NP
        """
        super(SpanStopPredictor, self).__init__()

        self.span_predictor = span_predictor

    def forward(self, x: Tensor) -> Tensor:
        """
        Projecting the input vectors x to 2 class prediction vector
        """
        idx = self.span_predictor(x)

        # Check if the predicted span is negative
        stop_span_prediction = (idx[:,0] < 0)

        # Create the binary stop signal
        out = torch.zeros_like(idx)
        if torch.cuda.is_available():
            out = out.cuda()

        # Set 1 to the stop class, 0 to the continue class, or vice verse
        out[stop_span_prediction, 1] = 1
        out[~stop_span_prediction, 0] = 1
        return out

