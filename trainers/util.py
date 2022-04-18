import torch
import numpy as np


def choose_span_sample(num_spans: int) -> int:
    possible_indices = np.arange(num_spans)
    # p = possible_indices + 1
    # p = p / np.sum(p)
    p = np.ones_like(possible_indices) / num_spans
    return np.random.choice(possible_indices, p=p)


def train_only_necassary_gpt2_weights(model: torch.nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad = False

    for i in range(12):
        model.gpt_encoder.h[i].attn.masked_bias.requires_grad = True

    return model
