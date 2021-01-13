import torch
from typing import Union, List
import torch_geometric

from utils.utils_functions import device, mae_fn, mse_fn


class LossFn:
    def __init__(self, w_e, w_f, w_q, w_p, action: Union[List[str], str] = "E"):
        # TODO return more detailed loss: MAE, MSE
        self.action = action
        self.w_e = w_e
        self.w_f = w_f
        self.w_q = w_q
        self.w_p = w_p

    def __call__(self, E_pred, F_pred, Q_pred, D_pred, data, requires_detail=False):
        if isinstance(self.action, list):
            # multi-task prediction
            assert E_pred.shape[-1] == len(self.action)
            tgt = torch.cat([getattr(data, name).view(-1, 1) for name in self.action], dim=-1)
            mae_loss = torch.mean(torch.abs(E_pred - tgt), dim=0, keepdim=True)
            rmse_loss = torch.sqrt(torch.mean((E_pred - tgt)**2, dim=0, keepdim=True))
            if requires_detail:
                detail = {"MAE_{}".format(name): mae_loss[:, i].item() for i, name in enumerate(self.action)}
                for i, name in enumerate(self.action):
                    detail["RMSE_{}".format(name)] = rmse_loss[:, i].item()
                return mae_loss.sum(), detail
            else:
                return mae_loss.sum()
        elif self.action == "E":
            # default PhysNet setting
            E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
            E_loss = self.w_e * torch.mean(torch.abs(E_pred - data.E))

            # if 'F' in data.keys():
            #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data['F'].to(device)),
            #                                                   data['atom_to_mol_batch'].to(device))
            #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

            Q_loss = self.w_q * torch.mean(torch.abs(Q_pred - data.Q))

            D_loss = self.w_p * torch.mean(torch.abs(D_pred - data.D))

            if requires_detail:
                return E_loss + F_loss + Q_loss + D_loss, {"MAE_E": E_loss.item(), "MAE_F": F_loss,
                                                           "MAE_Q": Q_loss.item(), "MAE_D": D_loss.item()}
            else:
                return E_loss + F_loss + Q_loss + D_loss
        elif self.action == "solubility":
            pass
        else:
            raise ValueError("Invalid action: {}".format(self.action))