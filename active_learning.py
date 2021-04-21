import argparse
import glob
import math
import shutil
from datetime import datetime

import os
import os.path as osp

import torch

from DataPrepareUtils import my_pre_transform, remove_atom_from_dataset
from Networks.PhysDimeNet import PhysDimeNet
from test import test_folder, test_step, test_info_analyze
from train import data_provider_solver, train, _add_arg_from_config
from utils.LossFn import LossFn
from utils.utils_functions import add_parser_arguments, preprocess_config, device, floating_type, collate_fn


def active_main():
    """
    The main function of active learning:
    --config_name: config file name
    --epoch_per_train: number of epoch per training cycle
    --n_cycle: number of cycles
    --top_percent: top n percent uncertainty molecules will be selected as training set
    :return:
    """
    # set up parser and arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    parser.add_argument('--epoch_per_train', default=10, type=int)  # TODO support list as input
    parser.add_argument('--n_cycle', default=10, type=int)
    parser.add_argument('--top_percent', default=10., type=float)
    parser.add_argument('--n_train_start', default=100, type=int)
    parser.add_argument('--random_sample', default='False', type=str)
    parser.add_argument('--new_mol_epoch', default=10, type=int)
    args = parser.parse_args()
    config_name = args.config_name

    config_args = parser.parse_args(['@{}'.format(config_name)])
    # this config is used in training added new molecules
    # TODO support parser
    epoch_per_train = config_args.epoch_per_train
    n_cycle = config_args.n_cycle
    top_percent = config_args.top_percent
    random_sample = (config_args.random_sample.lower() == 'true')
    new_mol_config = parser.parse_args(['@{}'.format(config_name)])
    new_mol_config.num_epochs = config_args.new_mol_epoch
    new_mol_config.ema_decay = 0.99

    debug_mode = (config_args.debug_mode.lower() == 'true')
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_dir = '{}_active_{}'.format(config_args.folder_prefix, current_time)
    if not osp.exists(run_dir):
        os.mkdir(run_dir)

    if config_args.data_provider != 'frag9to20_all':
        raise ValueError('Unsupported data_provider: {}'.format(config_args.data_provider))
    _kwargs = {'root': '../dataProviders/data', 'pre_transform': my_pre_transform, 'record_long_range': True,
               'type_3_body': 'B', 'cal_3body_term': True}
    dataset, _kwargs = data_provider_solver(config_args.data_provider, _kwargs)
    _kwargs = _add_arg_from_config(_kwargs, config_args)
    data_provider = dataset(**_kwargs)

    if debug_mode:
        print('WARNING, debug mode enabled!')
        train_index_total, val_index_total = data_provider.train_index, data_provider.val_index
        train_index_total = train_index_total[:1000]
    else:
        train_index_total, val_index_total, _ = remove_atom_from_dataset(5, data_provider, remove_split=('train', 'valid'))
    this_train_index = train_index_total[:args.n_train_start]

    # storing active options for future ref.
    with open(osp.join(run_dir, 'active_options.txt'), 'w') as f:
        _args_dict = vars(args)
        for key in _args_dict.keys():
            f.write('--{}={}\n'.format(key, _args_dict[key]))

    for cycle in range(n_cycle):
        config_args.folder_prefix = osp.join(run_dir, 'cycle{}'.format(cycle))
        if cycle > 0:
            # find latest trained folder for testing
            train_dir = glob.glob(osp.join(run_dir, 'cycle{}_run_*'.format(cycle-1)))[0]
        else:
            train_dir = glob.glob(config_args.use_trained_model)[0]
        net_kwargs = preprocess_config(config_args)
        net = PhysDimeNet(**net_kwargs).to(device).type(floating_type)
        net.load_state_dict(torch.load(osp.join(train_dir, 'best_model.pt')))
        w_e, w_f, w_q, w_p = 1, config_args.force_weight, config_args.charge_weight, config_args.dipole_weight
        loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p)
        if config_args.num_epochs != epoch_per_train:
            print('overwriting number_epochs from {} to {}'.format(config_args.num_epochs, epoch_per_train))
            config_args.num_epochs = epoch_per_train

        # test index = train total - this train
        # the following trick is to calculate set subtraction efficiently
        combined, count = torch.cat([this_train_index, train_index_total]).unique(return_counts=True)
        this_test_index = combined[count == 1]
        num_mol_add = math.ceil(len(this_test_index) * top_percent // 100)
        if random_sample:
            added_index = this_test_index[torch.randperm(len(this_test_index))[:num_mol_add]]
        else:
            # test and find large error molecules
            test_data_loader = torch.utils.data.DataLoader(
                data_provider[this_test_index], batch_size=32, collate_fn=collate_fn, pin_memory=device, shuffle=False)
            current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            test_dir = osp.join(run_dir, 'cycle{}_test_{}'.format(cycle, current_time))
            os.mkdir(test_dir)
            test_info, test_info_std = test_step(config_args, net, test_data_loader, len(this_test_index), loss_fn,
                                                 dataset_name='frag20_active', run_dir=test_dir, n_forward=5)
            test_info_analyze(23.061*test_info['E_pred'], 23.061*data_provider.data.E[this_test_index], test_dir)

            added_index = test_info_std['E_pred'].argsort(descending=True)[:num_mol_add]

        # train added index for 20 epochs with larger learning rate
        new_mol_config.folder_prefix = osp.join(run_dir, 'cycle{}_new_mol'.format(cycle))
        if cycle > 0:
            # last trained model location
            new_mol_config.use_trained_model = osp.join(run_dir, 'cycle{}_run*'.format(cycle-1))
        train(new_mol_config, data_provider, explicit_split=(added_index, val_index_total, None), ignore_valid=True)

        # train total for specified amount of epochs
        # last trained model is from trained new molecules
        config_args.use_trained_model = osp.join(run_dir, 'cycle{}_new_mol_*'.format(cycle))
        train(config_args, data_provider, explicit_split=(this_train_index, val_index_total, None), ignore_valid=True)

        torch.save(added_index, osp.join(train_dir, 'added_index.pt'))
        torch.save(this_test_index, osp.join(train_dir, 'reservoir_index.pt'))
        torch.save(this_train_index, osp.join(train_dir, 'this_train_index.pt'))
        this_train_index = torch.cat([this_train_index, added_index])


if __name__ == '__main__':
    active_main()
