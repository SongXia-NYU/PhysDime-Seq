import torch.nn as nn

from Networks.SharedLayers.ActivationFns import activation_getter
from Networks.UncertaintyLayers.MCDropout import ConcreteDropout
from utils.utils_functions import semi_orthogonal_glorot_weights, get_n_params


class ResidualLayer(nn.Module):
    """
    The residual layer defined in PhysNet
    """
    def __init__(self, F, activation, concrete_dropout=False):
        super().__init__()
        self.concrete_dropout = concrete_dropout
        self.activation = activation_getter(activation)

        self.lin1 = nn.Linear(F, F)
        self.lin1.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin1.bias.data.zero_()

        self.lin2 = nn.Linear(F, F)
        self.lin2.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin2.bias.data.zero_()

        if self.concrete_dropout:
            self.lin1 = ConcreteDropout(self.lin1, module_type='Linear')
            self.lin2 = ConcreteDropout(self.lin2, module_type='Linear')

    def forward(self, x):
        regularization = 0.
        x_res = x
        x = self.activation(x)
        if self.concrete_dropout:
            x, reg = self.lin1(x)
            regularization = reg + regularization
        else:
            x = self.lin1(x)
        x = self.activation(x)
        if self.concrete_dropout:
            x, reg = self.lin2(x)
            regularization = reg + regularization
            return x + x_res, regularization
        else:
            x = self.lin2(x)
            return x + x_res


if __name__ == '__main__':
    model = ResidualLayer(128)
    print(get_n_params(model))
