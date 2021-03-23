import torch
from typing import Union, List
from copy import deepcopy
import torch_geometric

from utils.utils_functions import device, mae_fn, mse_fn, kcal2ev


class LossFn:
    def __init__(self, w_e, w_f, w_q, w_p, action: Union[List[str], str] = "E", auto_sol=False, target_names=None):
        self.target_names = target_names
        self.action = deepcopy(action)
        self.w_e = w_e
        self.w_f = w_f
        self.w_q = w_q
        self.w_d = w_p
        self.auto_sol = auto_sol
        if self.auto_sol:
            assert "gasEnergy" in self.target_names
            if "watEnergy" in self.target_names:
                self.target_names.append("CalcSol")
            if "octEnergy" in self.target_names:
                self.target_names.append("CalcOct")

    def __call__(self, E_pred, F_pred, Q_pred, D_pred, data, loss_detail=False, diff_detail=False):
        if self.action in ["names", "names_and_QD"]:
            E_tgt, E_pred = self._get_target(E_pred, data)

            mae_loss = torch.mean(torch.abs(E_pred - E_tgt), dim=0, keepdim=True)
            rmse_loss = torch.sqrt(torch.mean((E_pred - E_tgt)**2, dim=0, keepdim=True))

            total_loss = mae_loss.sum()
            if loss_detail:
                detail = {"MAE_{}".format(name): mae_loss[:, i].item() for i, name in enumerate(self.target_names)}
                for i, name in enumerate(self.target_names):
                    detail["RMSE_{}".format(name)] = rmse_loss[:, i].item()
                    if diff_detail:
                        detail["DIFF_{}".format(name)] = [(E_pred - E_tgt)[:, i].detach().cpu().view(-1)]
            else:
                detail = None

            if self.action == "names_and_QD":
                q_mae = torch.mean(torch.abs(Q_pred - data.Q))
                d_mae = torch.mean(torch.abs(D_pred - data.D))
                total_loss = total_loss + self.w_q * q_mae + self.w_d * d_mae
                if loss_detail:
                    detail["MAE_Q"] = q_mae.item()
                    detail["MAE_D"] = d_mae.item()
                    if diff_detail:
                        detail["DIFF_Q"] = [(Q_pred - data.Q).detach().cpu().view(-1)]
                        detail["DIFF_D"] = [(D_pred - data.D).detach().cpu().view(-1)]

            if loss_detail:
                return total_loss, detail
            else:
                return total_loss

        elif self.action == "E":
            # default PhysNet setting
            E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
            E_loss = self.w_e * torch.mean(torch.abs(E_pred - data.E))

            # if 'F' in data.keys():
            #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data['F'].to(device)),
            #                                                   data['atom_to_mol_batch'].to(device))
            #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

            Q_loss = self.w_q * torch.mean(torch.abs(Q_pred - data.Q))

            D_loss = self.w_d * torch.mean(torch.abs(D_pred - data.D))

            if loss_detail:
                return E_loss + F_loss + Q_loss + D_loss, {"MAE_E": E_loss.item(), "MAE_F": F_loss,
                                                           "MAE_Q": Q_loss.item(), "MAE_D": D_loss.item()}
            else:
                return E_loss + F_loss + Q_loss + D_loss
        else:
            raise ValueError("Invalid action: {}".format(self.action))

    def _get_target(self, E_pred, data):
        """
        Get energy target from data
        Solvation energy is in kcal/mol but gas/water/octanol energy is in eV
        """
        # multi-task prediction
        if self.auto_sol:
            total_pred = [E_pred]
            gas_id = self.target_names.index("gasEnergy")
            for sol_name in ["watEnergy", "octEnergy"]:
                if sol_name in self.target_names:
                    sol_id = self.target_names.index(sol_name)
                    total_pred.append((E_pred[:, sol_id] - E_pred[:, gas_id]).view(-1, 1) / kcal2ev)
            E_pred = torch.cat(total_pred, dim=-1)

        assert E_pred.shape[-1] == len(self.target_names)
        E_tgt = torch.cat([getattr(data, name).view(-1, 1) for name in self.target_names], dim=-1)
        return E_tgt, E_pred
