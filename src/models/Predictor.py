from torch import nn


class Predictor(nn.Module):

    def __init__(self, input_dim, num_speakers):
        super(Predictor, self).__init__()
        self.out_layer = nn.Linear(input_dim, num_speakers)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.out_layer(x)
        probs = self.softmax(x)
        return probs
