import glob
import sys
import argparse
import logging
import math
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.cuda
import torch.utils.data
from warmup_scheduler import GradualWarmupScheduler

from CombinedIMDataset import CombinedIMDataset
from Frag9to20MixIMDataset import Frag9to20MixIMDataset, uniform_split, small_split, large_split
from Frag20IMDataset import Frag20IMDataset
from Networks.PhysDimeNet import PhysDimeNet
from DataPrepareUtils import my_pre_transform, remove_atom_from_dataset
from Networks.UncertaintyLayers.swag import SWAG
from PhysDimeIMDataset import PhysDimeIMDataset
from qm9InMemoryDataset import Qm9InMemoryDataset
from utils.Optimizers import EmaAmsGrad, MySGD
from utils.LossFn import LossFn
from utils.time_meta import print_function_runtime, record_data
from utils.utils_functions import device, add_parser_arguments, floating_type, kwargs_solver, get_lr, collate_fn, \
    load_data_from_index, get_n_params, cal_mean_std, print_val_results, remove_handler, option_solver


def data_provider_solver(name, _kw_args):
    """

    :param name:
    :param _kw_args:
    :return:
    """
    _options = option_solver(name)
    name = name.split('[')[0]
    for key in _options.keys():
        _kw_args[key] = _options[key]

    settings = name.split('+')[1:] if len(name.split('+')) > 1 else ['']
    if settings[0] == 'extBond':
        _kw_args['extended_bond'] = True
    name = name.split('+')[0]
    if name == 'qm9':
        return Qm9InMemoryDataset, _kw_args
    elif name.split('_')[0] == 'frag20nHeavy':
        n_heavy_atom = int(name[7:])
        _kw_args['n_heavy_atom'] = n_heavy_atom
        return Frag20IMDataset, _kw_args
    elif name[:9] == 'frag9to20':
        _kw_args['training_option'] = 'train'
        argument = name[10:]
        if argument == 'uniform':
            _kw_args['split_settings'] = uniform_split
        elif argument == 'small':
            _kw_args['split_settings'] = small_split
        elif argument == 'large':
            _kw_args['split_settings'] = large_split
        elif argument == 'all':
            _kw_args['split_settings'] = uniform_split
            _kw_args['all_data'] = True
        elif argument == 'jianing':
            _kw_args['split_settings'] = uniform_split
            _kw_args['jianing_split'] = True
        else:
            raise ValueError('not recognized argument: {}'.format(argument))
        return Frag9to20MixIMDataset, _kw_args
    elif name in ['frag20_eMol9_combine', 'frag20_eMol9_combine_MMFF']:
        geometry = "MMFF" if name == 'frag20_eMol9_combine_MMFF' else "QM"
        frag20dataset, tmp_args = data_provider_solver('frag9to20_all', _kw_args)
        frag20dataset = frag20dataset(**tmp_args, geometry=geometry)
        len_frag20 = len(frag20dataset)
        val_index = frag20dataset.val_index
        train_index = frag20dataset.train_index
        _kw_args['dataset_name'] = 'frag20_eMol9_combined_{}.pt'.format(geometry)
        _kw_args['val_index'] = val_index
        e_mol9_dataset = PhysDimeIMDataset(root=tmp_args['root'], processed_prefix='eMol9_{}'.format(geometry),
                                           pre_transform=my_pre_transform,
                                           record_long_range=tmp_args['record_long_range'],
                                           infile_dic={
                                               'PhysNet': 'eMol9_PhysNet_{}.npz'.format(geometry),
                                               'SDF': 'eMol9_{}.pt'.format(geometry)
                                           })
        len_e9 = len(e_mol9_dataset)
        _kw_args['train_index'] = torch.cat([train_index, torch.arange(len_frag20, len_e9+len_frag20)])
        _kw_args['dataset_list'] = [frag20dataset, e_mol9_dataset]
        return CombinedIMDataset, _kw_args
    else:
        raise ValueError('Unrecognized dataset name: {} !'.format(name))


def train_step(model, _optimizer, data_batch, loss_fn, max_norm, warm_up_scheduler):
    # torch.cuda.synchronize()
    # t0 = time.time()

    model.train()
    _optimizer.zero_grad()

    E_pred, F_pred, Q_pred, p_pred, loss_nh = model(data_batch)

    # t0 = record_data('forward', t0, True)

    loss = loss_fn(E_pred, F_pred, Q_pred, p_pred, data_batch) + loss_nh

    # print("Training b4 backward: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

    loss.backward()

    # print("Training after backward: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

    # t0 = record_data('backward', t0, True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    _optimizer.step()
    warm_up_scheduler.step()

    # t0 = record_data('step', t0, True)
    # print_function_runtime()

    result_loss = loss.data[0]

    return result_loss


def val_step(model, _data_loader, data_size, loss_fn, mae_fn, mse_fn, dataset_name='dataset', detailed_info=False,
             print_to_log=True):
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

        emae += _batch_size * mae_fn(E_pred, val_data.E.to(device)).item()
        emse += _batch_size * mse_fn(E_pred, val_data.E.to(device)).item()

        qmae += _batch_size * mae_fn(Q_pred, val_data.Q.to(device)).item()
        qmse += _batch_size * mse_fn(Q_pred, val_data.Q.to(device)).item()

        pmae += _batch_size * mae_fn(D_pred, val_data.D.to(device)).item()
        pmse += _batch_size * mse_fn(D_pred, val_data.D.to(device)).item()

        if detailed_info:
            for aggr, pred in zip([E_pred_aggr, Q_pred_aggr, D_pred_aggr], [E_pred, Q_pred, D_pred]):
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

    return result


def train(config_args, data_provider, explicit_split=None, ignore_valid=False):
    net_kwargs = kwargs_solver(config_args)

    debug_mode = (config_args.debug_mode.lower() == 'true')
    use_trained_model = (config_args.use_trained_model.lower() != 'false')
    use_swag = (config_args.uncertainty_modify.split('_')[0] == 'swag')

    # set up run directory
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_directory = config_args.folder_prefix + '_run_' + current_time
    os.mkdir(run_directory)

    shutil.copyfile(config_args.config_name, os.path.join(run_directory, config_args.config_name))

    # Logger setup
    logging.basicConfig(filename=os.path.join(run_directory, config_args.log_file_name),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Meta data file set up
    meta_data_name = os.path.join(run_directory, 'meta.txt')

    if explicit_split is not None:
        train_index, val_index, test_index = explicit_split
    else:
        train_index, val_index, test_index = data_provider.train_index, data_provider.val_index, data_provider.test_index
        for atom_id in config_args.remove_atom_ids:
            # remove B atom from dataset
            train_index, val_index, test_index = remove_atom_from_dataset(atom_id, data_provider, remove_split=('train', 'valid'),
                                                                          explicit_split=(train_index, val_index, test_index))
        logger.info('REMOVING ATOM {} FROM DATASET'.format(config_args.remove_atom_ids))
        print('REMOVING ATOM {} FROM DATASET'.format(config_args.remove_atom_ids))

    train_size = len(train_index)
    val_size = len(val_index)
    logger.info('train size: ' + str(train_size))
    logger.info('validation size: ' + str(val_size))
    num_train_batches = train_size // config_args.batch_size + 1

    train_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(train_index)], batch_size=config_args.batch_size, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(val_index)], batch_size=config_args.valid_batch_size, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), shuffle=True)

    mae_fn = torch.nn.L1Loss(reduction='mean')
    mse_fn = torch.nn.MSELoss(reduction='mean')
    w_e, w_f, w_q, w_p = 1, config_args.force_weight, config_args.charge_weight, config_args.dipole_weight
    loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p)

    # ###################################Setting up model and optimizer############################################# #
    # Normalization of PhysNet atom-wise prediction
    E_mean_atom, E_std_atom = cal_mean_std(data_provider.data.E, data_provider.data.N, train_index)
    E_atomic_scale = E_std_atom
    E_atomic_shift = E_mean_atom

    net_kwargs['energy_shift'] = E_atomic_shift
    net_kwargs['energy_scale'] = E_atomic_scale

    net = PhysDimeNet(**net_kwargs)
    shadow_net = PhysDimeNet(**net_kwargs)
    net = net.to(device)
    net = net.type(floating_type)
    shadow_net = shadow_net.to(device)
    shadow_net = shadow_net.type(floating_type)
    shadow_net.load_state_dict(net.state_dict())

    if use_swag:
        dummy_model = PhysDimeNet(**net_kwargs).to(device).type(floating_type)
        swag_model = SWAG(dummy_model, no_cov_mat=False, max_num_models=20)
    else:
        swag_model = None

    # retrain model
    if use_trained_model:
        trained_model_dir = glob.glob(config_args.use_trained_model)[0]
        logger.info('using trained model: {}'.format(config_args.use_trained_model))

        if os.path.exists(os.path.join(trained_model_dir, 'training_model.pt')):
            net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'training_model.pt'), map_location=device),
                                strict=False)
        else:
            net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pt'), map_location=device),
                                strict=False)
        shadow_net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pt'), map_location=device),
                                   strict=False)
    else:
        trained_model_dir = None

    # model freeze options (transfer learning)
    if config_args.freeze_option == 'prev':
        net.freeze_prev_layers(freeze_extra=False)
    elif config_args.freeze_option == 'prev_extra':
        net.freeze_prev_layers(freeze_extra=True)
    elif config_args.freeze_option == 'none':
        pass
    else:
        raise ValueError('Invalid freeze option: {}'.format(config_args.freeze_option))

    # optimizers
    if config_args.optimizer.split('_')[0] == 'emaAms':
        optimizer = EmaAmsGrad(net.parameters(), shadow_net, lr=config_args.learning_rate,
                               ema=float(config_args.optimizer.split('_')[1]))
    elif config_args.optimizer.split('_')[0] == 'sgd':
        optimizer = MySGD(net, lr=config_args.learning_rate)
    else:
        raise ValueError('Unrecognized optimizer: {}'.format(config_args.optimizer))
    if use_trained_model and (config_args.reset_optimizer.lower() == 'false'):
        if os.path.exists(os.path.join(trained_model_dir, "best_model_optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_optimizer.pt")))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config_args.decay_steps, gamma=0.1)
    warm_up_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=config_args.warm_up_steps,
                                               after_scheduler=scheduler) if config_args.warm_up_steps > 0 else scheduler

    # ###################################Printing meta data############################################# #
    if torch.cuda.is_available():
        logger.info('device name: ' + torch.cuda.get_device_name(device))
        logger.info("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

    with open(meta_data_name, 'w+') as f:
        n_parm, model_structure = get_n_params(net, None)
        logger.info('model params: {}'.format(n_parm))
        f.write('*' * 20 + '\n')
        f.write(model_structure)
        f.write('*' * 20 + '\n')
        f.write('train data index:{} ...\n'.format(train_index[:100]))
        f.write('val data index:{} ...\n'.format(val_index[:100]))
        # f.write('test data index:{} ...\n'.format(test_index[:100]))
        for _key in net_kwargs.keys():
            f.write("{} = {}\n".format(_key, net_kwargs[_key]))

    # ###################################Training############################################# #
    logger.info('start training...')

    shadow_net = optimizer.shadow_model
    val_loss = val_step(shadow_net, val_data_loader, val_size, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn)
    best_loss = val_loss['loss']

    logger.info('Init lr: {}'.format(get_lr(optimizer)))
    loss_data = []
    early_stop_count = 0
    for epoch in range(config_args.num_epochs):
        train_loss = 0.
        for batch_num, data in enumerate(train_data_loader):
            this_size = data.E.shape[0]

            train_loss += train_step(net, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
                                     max_norm=config_args.max_norm,
                                     warm_up_scheduler=warm_up_scheduler) * this_size / train_size

            if debug_mode & ((batch_num + 1) % 600 == 0):
                logger.info("Batch num: {}/{}, train loss: {} ".format(batch_num, num_train_batches, train_loss))

        logger.info('epoch {} ended, learning rate: {} '.format(epoch, get_lr(optimizer)))
        shadow_net = optimizer.shadow_model
        val_loss = val_step(shadow_net, val_data_loader, val_size, loss_fn=loss_fn, mae_fn=mae_fn, mse_fn=mse_fn)

        _loss_data_this_epoch = {'epoch': epoch,
                                 't_loss': train_loss,
                                 'v_loss': val_loss['loss'],
                                 'v_emae': val_loss['emae'],
                                 'v_ermse': val_loss['ermse'],
                                 'v_qmae': val_loss['qmae'],
                                 'v_qrmse': val_loss['qrmse'],
                                 'v_pmae': val_loss['pmae'],
                                 'v_prmse': val_loss['prmse'],
                                 'time': time.time()}
        loss_data.append(_loss_data_this_epoch)
        torch.save(loss_data, os.path.join(run_directory, 'loss_data.pt'))
        if use_swag:
            start, freq = config_args.uncertainty_modify.split('_')[1], config_args.uncertainty_modify.split('_')[2]
            if epoch > int(start) and (epoch % int(freq) == 0):
                swag_model.collect_model(shadow_net)
                torch.save(swag_model.state_dict(), os.path.join(run_directory, 'swag_model.pt'))
        if ignore_valid or (val_loss['loss'] < best_loss):
            early_stop_count = 0
            best_loss = val_loss['loss']
            torch.save(shadow_net.state_dict(), os.path.join(run_directory, 'best_model.pt'))
            torch.save(net.state_dict(), os.path.join(run_directory, 'training_model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(run_directory, 'best_model_optimizer.pt'))
        else:
            early_stop_count += 1
            if early_stop_count == config_args.early_stop:
                logger.info('early stop at epoch {}.'.format(epoch))
                break

    print_function_runtime(logger)
    remove_handler(logger)


def _add_arg_from_config(_kwargs, config_args):
    for attr_name in ['edge_version', 'cutoff', 'boundary_factor']:
        _kwargs[attr_name] = getattr(config_args, attr_name)
    return _kwargs


def main():
    # set up parser and arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)

    # parse config file
    config_name = 'config.txt'
    if len(sys.argv) == 1:
        if os.path.isfile(config_name):
            args, unknown = parser.parse_known_args(["@" + config_name])
        else:
            raise Exception('couldn\'t find \"config.txt\" !')
    else:
        args = parser.parse_args()
        config_name = args.config_name
        args, unknown = parser.parse_known_args(["@" + config_name])
    args.config_name = config_name

    _kwargs = {'root': '../dataProviders/data', 'pre_transform': my_pre_transform, 'record_long_range': True,
               'type_3_body': 'B', 'cal_3body_term': True}
    data_provider_class, _kwargs = data_provider_solver(args.data_provider, _kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    data_provider = data_provider_class(**_kwargs)
    train(args, data_provider)


if __name__ == "__main__":
    main()
