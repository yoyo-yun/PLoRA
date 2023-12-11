import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, config):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)


class UAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(UAttention, self).__init__()
        self.pre_pooling_linear_text = nn.Linear(input1_dim, config.pre_pooling_dim)
        self.pre_pooling_linear_attr = nn.Linear(input2_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, input_1, input_2, mask=None): # (input_1 is text representations, input_2 is attribute representation)
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear_text(input_1) + self.pre_pooling_linear_attr(input_2).unsqueeze(1))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        return torch.mul(input_1, weights.unsqueeze(2)).sum(dim=1)