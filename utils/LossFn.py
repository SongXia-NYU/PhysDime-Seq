import torch
import torch_geometric

from utils.utils_functions import device


class LossFn:
    def __init__(self, w_e, w_f, w_q, w_p):
        self.w_e = w_e
        self.w_f = w_f
        self.w_q = w_q
        self.w_p = w_p

    def __call__(self, E_pred, F_pred, Q_pred, D_pred, data):
        E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
        E_loss = self.w_e * torch.mean(torch.abs(E_pred - data.E))

        # if 'F' in data.keys():
        #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data['F'].to(device)),
        #                                                   data['atom_to_mol_batch'].to(device))
        #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

        Q_loss = self.w_q * torch.mean(torch.abs(Q_pred - data.Q))

        D_loss = self.w_p * torch.mean(torch.abs(D_pred - data.D))

        # /batch_size to normalize
        return E_loss + F_loss + Q_loss + D_loss
