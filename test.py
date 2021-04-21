import argparse
import math
from datetime import datetime
import logging
import os
import os.path as osp
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import shutil

from DataPrepareUtils import my_pre_transform, remove_atom_from_dataset
from Frag9to20MixIMDataset import Frag9to20MixIMDataset, uniform_split
from Networks.PhysDimeNet import PhysDimeNet
from Networks.UncertaintyLayers.swag import SWAG
from PhysDimeIMDataset import PhysDimeIMDataset
from PlatinumTestIMDataSet import PlatinumTestIMDataSet
from qm9InMemoryDataset import Qm9InMemoryDataset
from train import data_provider_solver, val_step_new, _add_arg_from_config
from deprecated_code.junk_code import val_step
from utils.LossFn import LossFn
from utils.utils_functions import add_parser_arguments, preprocess_config, device, floating_type, \
    collate_fn, remove_handler
from torch_scatter import scatter


def test_info_analyze(pred, target, test_dir, logger=None, name='Energy', threshold_base=1.0, unit='kcal/mol',
                      pred_std=None,
                      x_forward=0):
    diff = pred - target
    rank = torch.argsort(diff.abs())
    diff_ranked = diff[rank]
    if logger is None:
        logging.basicConfig(filename=os.path.join(test_dir, 'test.log'),
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        remove_logger = True
    else:
        remove_logger = False
    logger.info('Top 10 {} error: {}'.format(name, diff_ranked[-10:]))
    logger.info('Top 10 {} error id: {}'.format(name, rank[-10:]))
    e_mae = diff.abs().mean()
    logger.info('{} MAE: {}'.format(name, e_mae))
    thresholds = torch.logspace(-2, 2, 50) * threshold_base
    thresholds = thresholds.tolist()
    thresholds.extend([1.0, 10.])
    for threshold in thresholds:
        mask = (diff.abs() < threshold)
        logger.info('Percent of {} error < {:.4f} {}: {} out of {}, {:.2f}%'.format(
            name, threshold, unit, len(diff[mask]), len(diff), 100 * float(len(diff[mask])) / len(diff)))
    torch.save(diff, os.path.join(test_dir, 'diff.pt'))

    # concrete dropout
    if x_forward and (pred_std is not None):
        # print_scatter(pred_std, diff, name, unit, test_dir)
        print_uncetainty_figs(pred_std, diff, name, unit, test_dir)
    if x_forward:
        print_training_curve(test_dir)

    if remove_logger:
        remove_handler(logger)
    return


def get_test_set(args):
    default_kwargs = {'root': args["data_root"], 'pre_transform': my_pre_transform, 'record_long_range': True,
                      'type_3_body': 'B', 'cal_3body_term': True}
    dataset_cls, _kwargs = data_provider_solver(args["data_provider"], default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    dataset = dataset_cls(**_kwargs)
    print("used dataset: {}".format(dataset.processed_file_names))
    return dataset


def test_step(args, net, data_loader, total_size, loss_fn, mae_fn=torch.nn.L1Loss(reduction='mean'),
              mse_fn=torch.nn.MSELoss(reduction='mean'), dataset_name='data', run_dir=None,
              n_forward=50, **kwargs):
    if args["uncertainty_modify"] == 'none':
        result = val_step_new(net, data_loader, loss_fn, is_testing=True)
        torch.save(result, os.path.join(run_dir, 'loss_{}.pt'.format(dataset_name)))
        return result, None
    elif args["uncertainty_modify"].split('_')[0].split('[')[0] in ['concreteDropoutModule', 'concreteDropoutOutput',
                                                                    'swag']:
        print("You need to update the code of val_step_new")
        if os.path.exists(os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward))):
            print('loading exist files!')
            avg_result = torch.load(os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward)))
            std_result = torch.load(os.path.join(run_dir, dataset_name + '-std{}.pt'.format(n_forward)))
        else:
            avg_result = {}
            std_result = {}
            cum_result = {'E_pred': [], 'D_pred': [], 'Q_pred': []}
            for i in range(n_forward):
                if args["uncertainty_modify"].split('_')[0] == 'swag':
                    net.sample(scale=1.0, cov=True)
                result_i = val_step(net, data_loader, total_size, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn,
                                    dataset_name=dataset_name, print_to_log=False, detailed_info=True, **kwargs)

                for key in cum_result.keys():
                    cum_result[key].append(result_i[key])
            # list -> tensor
            for key in cum_result.keys():
                cum_result[key] = torch.stack(cum_result[key])
                avg_result[key] = cum_result[key].mean(dim=0)
                std_result[key] = cum_result[key].std(dim=0)
            torch.save(avg_result, os.path.join(run_dir, dataset_name + '-avg{}.pt'.format(n_forward)))
            torch.save(std_result, os.path.join(run_dir, dataset_name + '-std{}.pt'.format(n_forward)))
        return avg_result, std_result
    else:
        raise ValueError('unrecognized uncertainty_modify: {}'.format(args.uncertainty_modify))


def test_folder(folder_name, n_forward, x_forward, use_exist=False, check_active=False):
    # --------------- Dealing with active learning and uncertainty models -------------------- #
    if folder_name.find('active') >= 0 and check_active:
        cycle_folders = glob.glob(osp.join(folder_name, 'cycle*_run_*'))
        for ele in cycle_folders:
            # removing new molecules training folder
            if ele.find('_new_mol_') > 0:
                cycle_folders.remove(ele)
        cycle_folders.sort(key=lambda s: int(s.split('_')[0][5:]))
        folder_name = cycle_folders[-1]
        print('testing active learning folder: {}'.format(folder_name))
    # parse config file
    if osp.exists(osp.join(folder_name, "config-test.txt")):
        config_name = osp.join(folder_name, "config-test.txt")
    else:
        config_name = glob.glob(osp.join(folder_name, 'config-exp*.txt'))[0]
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    args, unknown = parser.parse_known_args(["@" + config_name])
    inferred_prefix = folder_name.split('_run_')[0]
    if args.folder_prefix != inferred_prefix:
        print('overwriting folder {} ----> {}'.format(args.folder_prefix, inferred_prefix))
        args.folder_prefix = inferred_prefix
    use_swag = (args.uncertainty_modify.split('_')[0] == 'swag')

    args = vars(args)
    args = preprocess_config(args)
    args["requires_embedding"] = True
    net = PhysDimeNet(**args)
    net = net.to(device)
    net = net.type(floating_type)

    if use_swag:
        net = SWAG(net, no_cov_mat=False, max_num_models=20)
        model_data = torch.load(os.path.join(folder_name, 'swag_model.pt'), map_location=device)
    else:
        model_data = torch.load(os.path.join(folder_name, 'best_model.pt'), map_location=device)

    net.load_state_dict(model_data)
    w_e, w_f, w_q, w_p = 1, args["force_weight"], args["charge_weight"], args["dipole_weight"]

    # why did I do that?
    # action = args.target_names if args.action != "E" else "E"
    loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, action=args["action"],
                     auto_sol=("gasEnergy" in args["target_names"]),
                     target_names=args["target_names"])

    mae_fn = torch.nn.L1Loss(reduction='mean')
    mse_fn = torch.nn.MSELoss(reduction='mean')

    test_prefix = args["folder_prefix"] + '_test_'

    if use_exist:
        test_dir = glob.glob('{}*'.format(test_prefix))[0]
    else:
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        folder_dir = osp.dirname(folder_name)
        test_dir = test_prefix + current_time
        test_dir = osp.join(folder_dir, test_dir)
        os.mkdir(test_dir)

    shutil.copyfile(os.path.join(folder_name, 'loss_data.pt'), os.path.join(test_dir, 'loss_data.pt'))
    shutil.copy(config_name, test_dir)

    logging.basicConfig(filename=os.path.join(test_dir, "test.log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("dataset in args: {}".format(args["data_provider"]))

    # ---------------------------------- Testing -------------------------------- #
    data_provider = get_test_set(args)
    if isinstance(data_provider, tuple):
        data_provider_test = data_provider[1]
        data_provider = data_provider[0]
    else:
        data_provider_test = data_provider

    if args["remove_atom_ids"] > 0:
        _, data_provider.val_index, _ = remove_atom_from_dataset(args["remove_atom_ids"], data_provider, ("valid",),
                                                                 (None, data_provider.val_index, None))
        _, _, data_provider.test_index = remove_atom_from_dataset(args["remove_atom_ids"], data_provider_test,
                                                                  ('test',),
                                                                  (None, None, data_provider.test_index))

    logger.info("dataset: {}".format(data_provider.processed_file_names))
    logger.info("valid size: {}".format(len(data_provider.val_index)))
    logger.info("test size: {}".format(len(data_provider.test_index)))

    for index_name in ["train_index", "val_index", "test_index"]:
        index_name_ = index_name.split("_")[0]
        this_index = getattr(data_provider, index_name)
        this_data_loader = torch.utils.data.DataLoader(
            data_provider[torch.as_tensor(this_index)], batch_size=args["valid_batch_size"], collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(), shuffle=False)
        this_info, this_std = test_step(args, net, this_data_loader, len(this_index), loss_fn=loss_fn, mae_fn=mae_fn,
                                        mse_fn=mse_fn, dataset_name='{}_{}'.format(args["data_provider"], index_name_),
                                        run_dir=test_dir, n_forward=n_forward, action=args["action"])
        logger.info("-------------- {} ---------------".format(index_name_))
        for key in this_info:
            logger.info("{}: {}".format(key, this_info[key]))
        logger.info("----------- end of {} ------------".format(index_name_))

    # if not os.path.exists(os.path.join(test_directory, 'loss.pt')):
    #     loss = cal_loss(test_info, data_provider_test.data.E[test_index], data_provider_test.data.D[test_index],
    #                     data_provider_test.data.Q[test_index], mae_fn=mae_fn, mse_fn=mse_fn)
    #     torch.save(loss, os.path.join(test_directory, 'loss.pt'))

    # remove global variables
    remove_handler(logger)
    del parser
    del args


def cal_loss(pred, e_target, d_target, q_target, mae_fn, mse_fn):
    result = {
        'loss': None,
        'emae': mae_fn(pred['E_pred'], e_target),
        'ermse': torch.sqrt(mse_fn(pred['E_pred'], e_target)),
        'pmae': mae_fn(pred['D_pred'], d_target),
        'prmse': torch.sqrt(mse_fn(pred['D_pred'], d_target)),
        'qmae': mae_fn(pred['Q_pred'], q_target),
        'qrmse': torch.sqrt(mse_fn(pred['Q_pred'], q_target)),
    }
    return result


def test_all():
    parser = argparse.ArgumentParser()
    # parser = add_parser_arguments(parser)
    parser.add_argument('--folder_names', default='../PhysDimeTestTmp/exp*_run_*', type=str)
    parser.add_argument('--x_forward', default=1, type=int)
    parser.add_argument('--n_forward', default=25, type=int)
    parser.add_argument('--use_exist', action="store_true")
    _args = parser.parse_args()

    run_dirs = glob.glob(_args.folder_names)

    for name in run_dirs:
        print('testing folder: {}'.format(name))
        test_folder(name, _args.n_forward, _args.x_forward, _args.use_exist)


def print_uncetainty_figs(pred_std, diff, name, unit, test_dir, n_bins=10):
    # let x axis ordered ascending
    std_rank = torch.argsort(pred_std)
    pred_std = pred_std[std_rank]
    diff = diff[std_rank]

    diff = diff.abs()
    diff_2 = diff ** 2
    x_data = torch.arange(pred_std.min(), pred_std.max(), (pred_std.max() - pred_std.min()) / n_bins)
    mae_data = torch.zeros(x_data.shape[0] - 1).float()
    rmse_data = torch.zeros_like(mae_data)
    for i in range(x_data.shape[0] - 1):
        mask = (pred_std < x_data[i + 1]) & (pred_std > x_data[i])
        mae_data[i] = diff[mask].mean()
        rmse_data[i] = torch.sqrt(diff_2[mask].mean())

    plt.figure(figsize=(15, 10))
    # Plotting predicted error MAE vs uncertainty
    plt.plot(x_data[1:], mae_data, label='{} MAE, {}'.format(name, unit))
    plt.plot(x_data[1:], rmse_data, label='{} RMSE, {}'.format(name, unit))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.savefig(os.path.join(test_dir, 'avg_error_uncertainty'))

    fig, ax1 = plt.subplots(figsize=(15, 10))
    diff_abs = diff.abs()
    ax1.scatter(pred_std, diff_abs, alpha=0.1)
    ax1.set_xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax1.set_ylabel('Error of {}, {}'.format(name, unit))
    # plt.title('Uncertainty vs. prediction error')
    # plt.savefig(os.path.join(test_dir, 'uncertainty'))

    # Plotting cumulative large error percent vs uncertainty
    thresholds = ['0', '1.0', '10.0']
    cum_large_count = {threshold: torch.zeros_like(x_data) for threshold in thresholds}
    for i in range(x_data.shape[0]):
        mask = (pred_std < x_data[i])
        select_diff = diff[mask]
        for threshold in thresholds:
            cum_large_count[threshold][i] = select_diff[select_diff > float(threshold)].shape[0]
    # plt.figure(figsize=(15, 10))
    ax2 = ax1.twinx()
    for threshold in thresholds:
        count = cum_large_count[threshold]
        ax2.plot(x_data, count / count[-1] * 100,
                 label='%all molecules' if threshold == '0' else '%large>{}kcal/mol'.format(threshold))

    plt.legend()
    # ax2.xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax2.set_ylabel('percent of molecules')
    plt.savefig(os.path.join(test_dir, 'percent'))

    # Plotting density of large error percent vs uncertainty
    x_mid = (x_data[:-1] + x_data[1:]) / 2
    plt.figure(figsize=(15, 10))
    for i, threshold in enumerate(thresholds):
        count_density_all = cum_large_count['0'][1:] - cum_large_count['0'][:-1]
        count_density = cum_large_count[threshold][1:] - cum_large_count[threshold][:-1]
        count_density_lower = count_density_all - count_density
        width = (x_data[1] - x_data[0]) / (len(thresholds) * 5)
        plt.bar(x_mid + i * width, count_density / count_density.sum() * 100, width=width,
                label='all molecules' if threshold == '0' else 'large>{}kcal/mol'.format(threshold))
        # if threshold != '0':
        #     plt.bar(x_mid - i * width, count_density_lower / count_density_lower.sum() * 100, width=width,
        #             label='large<{}kcal/mol'.format(threshold))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('density of molecules')
    plt.xticks(x_data)
    plt.savefig(os.path.join(test_dir, 'percent_density'))

    # box plot
    num_points = diff.shape[0]
    step = math.ceil(num_points / n_bins)
    sep_index = torch.arange(0, num_points + 1, step)
    y_blocks = []
    x_mean = []
    for num, start in enumerate(sep_index):
        if num + 1 < sep_index.shape[0]:
            end = sep_index[num + 1]
        else:
            end = num_points
        y_blocks.append(diff[start: end].numpy())
        x_mean.append(pred_std[start: end].mean().item())
    plt.figure(figsize=(15, 10))
    box_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.boxplot(y_blocks, notch=True, positions=x_mean, vert=True, showfliers=False,
                widths=box_size)
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.title('Boxplot')
    plt.savefig(os.path.join(test_dir, 'box_plot'))

    # interval percent
    small_err_percent_1 = np.asarray([(x < 1.).sum() / x.shape[0] for x in y_blocks])
    large_err_percent_10 = np.asarray([(x > 10.).sum() / x.shape[0] for x in y_blocks])
    x_mean = np.asarray(x_mean)
    plt.figure(figsize=(15, 10))
    bar_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.bar(x_mean - bar_size, large_err_percent_10, label='percent, error > 10kcal/mol', width=bar_size)
    plt.bar(x_mean, 1 - large_err_percent_10 - small_err_percent_1,
            label='percent, 1kcal/mol < error < 10kcal/mol', width=bar_size)
    plt.bar(x_mean + bar_size, small_err_percent_1, label='small error < 1kcal/mol', width=bar_size)
    plt.legend()
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('percent')
    plt.title('error percent')
    plt.savefig(os.path.join(test_dir, 'error_percent'))
    return


if __name__ == "__main__":
    test_all()
