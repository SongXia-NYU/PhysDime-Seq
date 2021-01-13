import logging
import math
from typing import Union, List

import torch

from utils.utils_functions import floating_type, device, print_val_results


def val_step(model, _data_loader, data_size, loss_fn, mae_fn, mse_fn, dataset_name='dataset', detailed_info=False,
             print_to_log=True, action: Union[List[str], str] = "E"):
    print("this valid step is deprecated_code.")
    model.eval()
    loss, emae, emse, fmae, fmse, qmae, qmse, pmae, pmse = 0, 0, 0, 0, 0, 0, 0, 0, 0

    if detailed_info:
        E_pred_aggr = torch.zeros(data_size).cpu().type(floating_type)
        Q_pred_aggr = torch.zeros_like(E_pred_aggr)
        D_pred_aggr = torch.zeros(data_size, 3).cpu().type(floating_type)
        idx_before = 0
    else:
        E_pred_aggr, Q_pred_aggr, D_pred_aggr, idx_before = None, None, None, None

    for val_data in _data_loader:
        _batch_size = len(val_data.E)

        E_pred, F_pred, Q_pred, D_pred, loss_nh = model(val_data)

        # IMPORTANT the .item() function is necessary here, otherwise the graph will be unintentionally stored and
        # never be released
        # And you will run out of memory after several val()
        loss1 = loss_fn(E_pred, F_pred, Q_pred, D_pred, val_data).item()
        loss2 = loss_nh.item()
        loss += _batch_size * (loss1 + loss2)

        emae += _batch_size * mae_fn(E_pred, getattr(val_data, action).to(device)).item()
        emse += _batch_size * mse_fn(E_pred, getattr(val_data, action).to(device)).item()

        if Q_pred is not None:
            qmae += _batch_size * mae_fn(Q_pred, val_data.Q.to(device)).item()
            qmse += _batch_size * mse_fn(Q_pred, val_data.Q.to(device)).item()

            pmae += _batch_size * mae_fn(D_pred, val_data.D.to(device)).item()
            pmse += _batch_size * mse_fn(D_pred, val_data.D.to(device)).item()

        if detailed_info:
            for aggr, pred in zip([E_pred_aggr, Q_pred_aggr, D_pred_aggr], [E_pred, Q_pred, D_pred]):
                if pred is not None:
                    aggr[idx_before: _batch_size + idx_before] = pred.detach().cpu()
                # aggr[idx_before: _batch_size + idx_before] = val_data.E.detach().cpu().view(-1)
            idx_before = idx_before + _batch_size

    loss = loss / data_size
    emae = emae / data_size
    ermse = math.sqrt(emse / data_size)
    qmae = qmae / data_size
    qrmse = math.sqrt(qmse / data_size)
    pmae = pmae / data_size
    prmse = math.sqrt(pmse / data_size)

    if print_to_log:
        log_info = print_val_results(dataset_name, loss, emae, ermse, qmae, qrmse, pmae, prmse)
        logging.info(log_info)

    result = {'loss': loss, 'emae': emae, 'ermse': ermse, 'qmae': qmae, 'qrmse': qrmse, 'pmae': pmae, 'prmse': prmse}

    if detailed_info:
        result['E_pred'] = E_pred_aggr
        result['Q_pred'] = Q_pred_aggr
        result['D_pred'] = D_pred_aggr

    return
