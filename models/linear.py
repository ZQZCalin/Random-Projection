import torch.nn as nn
import math

class ApproximateLinear(nn.Module):
    def __init__(self, in_features, out_features, reduced_dim, bias=True) -> None:
        super(ApproximateLinear, self).__init__()

        # $\E[P^TP]_{ii} = \sum_{j=1}^s \E[p_{ji}^2] = 1$
        # ==> std = 1/sqrt(reduced_dim)
        self.std = 1 / math.sqrt(reduced_dim)

        # we only need the weight matrix from original layer
        # because bias vector is not affected by dimension reduction
        self.benchmark = nn.Linear(in_features, out_features, bias=False)
        self.projection = nn.Linear(in_features, reduced_dim, bias=False)
        self.approximate = nn.Linear(reduced_dim, out_features, bias)

        self.benchmark.weight.requires_grad_(False)
        self.projection.weight.requires_grad_(False)

        self.approximate_param_ghost = self.approximate.weight.data
        self.resample()

    def forward(self, x):
        return self.approximate(self.projection(x))

    def resample(self, std=None):
        # 1. update benchmark layer
        # project changes of approximate layer back to original space
        self.benchmark.weight.add_(
            (self.approximate.weight.data - self.approximate_param_ghost) 
            @ self.projection.weight.data)
        # update ghost data
        self.approximate_param_ghost.copy_(self.approximate.weight.data)

        # 2. resample gaussian layer
        std = std if std else self.std
        nn.init.normal_(self.projection.weight, mean=0., std=std)

        # 3. update approximate layer
        # project benchmark layer to subspace
        self.approximate.weight.data.copy_(
            self.benchmark.weight.data @ self.projection.weight.T)


# testing code
if __name__ == '__main__':
    net = ApproximateLinear(in_features=2, out_features=1, reduced_dim=3)
    # print(net.approximate.bias)
    print(net.state_dict())
    # print(net.approximate.weight.data)

