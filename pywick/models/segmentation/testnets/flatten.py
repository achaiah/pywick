from torch.nn import Module


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
