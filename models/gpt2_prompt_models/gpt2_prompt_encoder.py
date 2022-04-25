from typing import Dict

from torch import nn, Tensor
from transformers import GPT2Tokenizer, GPT2Model


class PromptEncoder(nn.Module):
    def __init__(self, prompt_reduction: nn.Module, pretraining: str = 'gpt2'):
        """
        prompt_redeuction - a module mapping the n*h hidden states (a hidden state for each token)
            -> a single 1*h vector (single vector representing the phrase
        pretrainign - a string stating the type of pretrained weights for GPT-2
        """
        super(PromptEncoder, self).__init__()
        self.gpt_encoder = GPT2Model.from_pretrained(pretraining)
        self.prompt_reduction = prompt_reduction

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        h = self.gpt_encoder(**x).last_hidden_state  # Get that last hidden state of all
        h = self.prompt_reduction(h)  # Get a single vector representing the entire prompt
        return h
