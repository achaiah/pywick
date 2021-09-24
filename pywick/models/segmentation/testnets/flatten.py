from torch.nn import Module


class Flatten(Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)
