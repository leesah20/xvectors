__author__ = "Lisa van Staden"

from torch import nn
from torch.nn import functional as F


class TDNN(nn.Module):

    def __init__(self, input_dim, output_dim, context_size, dilation=1, use_batch_norm=True):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.use_batch_norm = use_batch_norm
        self.linear = nn.Linear(input_dim * context_size, output_dim)
        self.relu = nn.ReLU()

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        batch_size, seq_len, d = x.size()
        x = x.unsqueeze(1)
        x = F.unfold(x, kernel_size=(self.context_size, d), dilation=(self.dilation, 1), stride=(1, d))

        x = x.transpose(1, 2)
        x = self.linear(x)
        x = self.relu(x)
        return x


