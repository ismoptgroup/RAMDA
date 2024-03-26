import torch

class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = torch.nn.Linear(in_features=784, out_features=10)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.log_softmax(x)
