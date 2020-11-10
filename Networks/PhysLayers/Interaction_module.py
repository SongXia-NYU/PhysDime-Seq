import torch
import torch.nn as nn
import torch.nn.functional
import torch_geometric

from utils.utils_functions import semi_orthogonal_glorot_weights, floating_type
from Networks.SharedLayers.ResidualLayer import ResidualLayer
from Networks.SharedLayers.ActivationFns import activation_getter


class InteractionModule(nn.Module):
    """
    The interaction layer defined in PhysNet
    """

    def __init__(self, F, K, n_res_interaction, activation):
        super().__init__()
        u = torch.Tensor(1, F).type(floating_type).fill_(1.)
        self.register_parameter('u', torch.nn.Parameter(u, True))

        self.message_pass_layer = MessagePassingLayer(aggr='add', F=F, K=K, activation=activation)

        self.n_res_interaction = n_res_interaction
        for i in range(n_res_interaction):
            self.add_module('res_layer' + str(i), ResidualLayer(F=F, activation=activation))

        self.lin_last = nn.Linear(F, F)
        self.lin_last.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_last.bias.data.zero_()

        self.activation = activation_getter(activation)

    def forward(self, x, edge_index, edge_attr):
        msged_x = self.message_pass_layer(x, edge_index, edge_attr)
        tmp_res = msged_x
        for i in range(self.n_res_interaction):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)
        v = self.activation(tmp_res)
        v = self.lin_last(v)
        return v + torch.mul(x, self.u), msged_x


class MessagePassingLayer(torch_geometric.nn.MessagePassing):
    """
    message passing layer in torch_geometric
    see: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html for more details
    """

    def __init__(self, F, K, activation, aggr='add', flow='source_to_target',):
        super().__init__(aggr=aggr, flow=flow)
        self.lin_for_same = nn.Linear(F, F)
        self.lin_for_same.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_same.bias.data.zero_()

        self.lin_for_diff = nn.Linear(F, F)
        self.lin_for_diff.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_diff.bias.data.zero_()

        self.G = nn.Linear(K, F, bias=False)
        self.G.weight.data.zero_()

        self.activation = activation_getter(activation)

    def message(self, x_j, edge_attr):
        msg = self.lin_for_diff(x_j)
        msg = self.activation(msg)
        masked_edge_attr = self.G(edge_attr)
        msg = torch.mul(msg, masked_edge_attr)
        return msg

    def forward(self, x, edge_index, edge_attr):
        x = self.activation(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def update(self, aggr_out, x):
        a = self.activation(self.lin_for_same(x))
        return a + aggr_out


if __name__ == '__main__':
    """
    Testing messaging passing layer
    make sure interaction modules are correct
    """
    input_h = torch.randn(10, 128)
    input_g = torch.Tensor(4, 64).type(floating_type).uniform_(0.1, 6)
    input_g[1] = input_g[0]
    input_g[2] = input_g[3]
    input_edge_index = torch.LongTensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    test_msg_layer = MessagePassingLayer()
    # test_msg_layer.G.weight.data = torch.zeros_like(test_msg_layer.G.weight.data)
    out = test_msg_layer(input_h, input_edge_index, input_g)
    activated_h = current_activation_fn(input_h)

    estimated = test_msg_layer.lin_for_same(activated_h)
    estimated = current_activation_fn(estimated)

    interaction = test_msg_layer.lin_for_diff(activated_h[1])
    interaction = current_activation_fn(interaction)
    interaction = torch.mul(interaction, test_msg_layer.G(input_g[0]))
    estimated[0] = estimated[0] + interaction

    interaction = test_msg_layer.lin_for_diff(activated_h[2])
    interaction = current_activation_fn(interaction)
    interaction = torch.mul(interaction, test_msg_layer.G(input_g[2]))
    estimated[0] = estimated[0] + interaction
    print('output:')
    print(out)
    print('-' * 30)
    print('estimated output')
    print(estimated)
    print('-' * 30)
    diff_part = (out[0] != estimated[0])
    print('Different part:')
    print(out[0, diff_part])
    print(estimated[0, diff_part])
    print(torch.sum(torch.abs(out - estimated), dim=1))
