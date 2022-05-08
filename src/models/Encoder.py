__author__ = "Lisa van Staden"

import torch
from torch import nn
from models.TDNN import TDNN


class Encoder(nn.Module):
    def __init__(self, num_frames, context_sizes, dilations, input_dim, output_dim, hidden_dim, segment_layer_dim):
        super(Encoder, self).__init__()

        self.tdnn_layers = nn.ModuleList([TDNN(context_size=context_sizes[i],
                                               input_dim=input_dim if i == 0 else hidden_dim,
                                               output_dim=hidden_dim if i < (num_frames - 1) else output_dim,
                                               dilation=dilations[i])
                                          for i in range(num_frames)])

        self.segment_layer1 = nn.Linear(output_dim * 2, segment_layer_dim)
        self.segment_layer2 = nn.Linear(segment_layer_dim, segment_layer_dim)

    def forward(self, x):
        for layer in self.tdnn_layers:
            x = layer(x)
        x = torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1)
        x = self.segment_layer1(x)
        return self.segment_layer2(x)
