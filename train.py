import glob
import sys
import argparse
import logging
import os
import os.path as osp
import shutil
import time
from datetime import datetime

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
from utils.time_meta import print_function_runtime
from utils.utils_functions import device, add_parser_arguments, floating_type, kwargs_solver, get_lr, collate_fn, \
    get_n_params, atom_mean_std, remove_handler, option_solver

default_kwargs = {'root': '../dataProviders/data', 'pre_transform': my_pre_transform, 'record_long_range': True,
                  'type_3_body': 'B', 'cal_3body_term': True}


def data_provider_solver(name_full, _kw_args):
    """

    :param name_full: Name should be in a format: ${name_base}[${key}=${value}], all key-value pairs will be feed into
    data_provider **kwargs
    :param _kw_args:
    :return: Data Provider Class and kwargs
    """
    additional_kwargs = option_solver(name_full)
    for key in additional_kwargs.keys():
        '''Converting string into corresponding data type'''
        if additional_kwargs[key] in ["True", "False"]:
            additional_kwargs[key] = (additional_kwargs[key] == "True")
        else:
            try:
                additional_kwargs[key] = float(additional_kwargs[key])
            except ValueError:
                pass
    name_base = name_full.split('[')[0]
    for key in additional_kwargs.keys():
        _kw_args[key] = additional_kwargs[key]

    if name_base == 'qm9':
        return Qm9InMemoryDataset, _kw_args
    elif name_base.split('_')[0] == 'frag20nHeavy':
        print("Deprecated dataset: {}".format(name_base))
        n_heavy_atom = int(name_base[7:])
        _kw_args['n_heavy_atom'] = n_heavy_atom
        return Frag20IMDataset, _kw_args
    elif name_base[:9] == 'frag9to20':
        _kw_args['training_option'] = 'train'
        split = name_base[10:]
        if split == 'uniform':
            _kw_args['split_settings'] = uniform_split
        elif split == 'small':
            _kw_args['split_settings'] = small_split
        elif split == 'large':
            _kw_args['split_settings'] = large_split
        elif split == 'all':
            _kw_args['split_settings'] = uniform_split
            _kw_args['all_data'] = True
        elif split == 'jianing':
            _kw_args['split_settings'] = uniform_split
            _kw_args['jianing_split'] = True
        else:
            raise ValueError('not recognized argument: {}'.format(split))
        return Frag9to20MixIMDataset, _kw_args
    elif name_base in ['frag20_eMol9_combine', 'frag20_eMol9_combine_MMFF']:
        geometry = "MMFF" if name_base == 'frag20_eMol9_combine_MMFF' else "QM"
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
        _kw_args['train_index'] = torch.cat([train_index, torch.arange(len_frag20, len_e9 + len_frag20)])
        _kw_args['dataset_list'] = [frag20dataset, e_mol9_dataset]
        return CombinedIMDataset, _kw_args
    else:
        raise ValueError('Unrecognized dataset name: {} !'.format(name_base))


def train_step(model, _optimizer, data_batch, loss_fn, max_norm, warm_up_scheduler):
    # torch.cuda.synchronize()
    # t0 = time.time()
    with torch.autograd.set_detect_anomaly(True):
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


def val_step_new(model, _data_loader, loss_fn):
    model.eval()
    valid_size = 0
    loss = 0.
    detail = None
    for val_data in _data_loader:
        _batch_size = len(val_data.E)
        E_pred, F_pred, Q_pred, D_pred, loss_nh = model(val_data)
        aggr_loss, loss_detail = loss_fn(E_pred, F_pred, Q_pred, D_pred, val_data, requires_detail=True)
        loss += aggr_loss.item() * _batch_size
        valid_size += _batch_size
        if detail is None:
            detail = loss_detail
        else:
            for key in detail:
                detail[key] += loss_detail[key] * _batch_size
    loss /= valid_size
    for key in detail:
        detail[key] /= valid_size
    detail["loss"] = loss
    return detail


def train(config_args, data_provider, explicit_split=None, ignore_valid=False):
    # ------------------- variable set up ---------------------- #
    net_kwargs = kwargs_solver(config_args)
    config_dict = vars(config_args)
    for bool_key in ["debug_mode", "use_trained_model", "auto_sol", "reset_optimizer"]:
        config_dict[bool_key] = (config_dict[bool_key].lower() != "false")
    config_dict["use_swag"] = (config_dict["uncertainty_modify"].split('_')[0] == 'swag')

    # ----------------- set up run directory -------------------- #
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        run_directory = config_dict["folder_prefix"] + '_run_' + current_time
        if not os.path.exists(run_directory):
            os.mkdir(run_directory)
            break
        else:
            time.sleep(10)

    shutil.copyfile(config_dict["config_name"], os.path.join(run_directory, config_dict["config_name"]))

    # --------------------- Logger setup ---------------------------- #
    logging.basicConfig(filename=os.path.join(run_directory, config_dict["log_file_name"]),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # -------------------- Meta data file set up -------------------- #
    meta_data_name = os.path.join(run_directory, 'meta.txt')

    # -------------- Index file and remove specific atoms ------------ #
    if explicit_split is not None:
        train_index, val_index, test_index = explicit_split
    else:
        train_index, val_index, test_index = data_provider.train_index, data_provider.val_index, data_provider.test_index
        if config_dict["remove_atom_ids"] > 0:
            # remove B atom from dataset
            train_index, val_index, test_index = remove_atom_from_dataset(config_dict["remove_atom_ids"], data_provider,
                                                                          remove_split=('train', 'valid'),
                                                                          explicit_split=(
                                                                              train_index, val_index, test_index))
        logger.info('REMOVING ATOM {} FROM DATASET'.format(config_dict["remove_atom_ids"]))
        print('REMOVING ATOM {} FROM DATASET'.format(config_dict["remove_atom_ids"]))

    train_size = len(train_index)
    val_size = len(val_index)
    logger.info('train size: ' + str(train_size))
    logger.info('validation size: ' + str(val_size))
    num_train_batches = train_size // config_dict["batch_size"] + 1

    train_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(train_index)], batch_size=config_dict["batch_size"], collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(val_index)], batch_size=config_dict["valid_batch_size"], collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), shuffle=True)

    w_e, w_f, w_q, w_p = 1., config_dict["force_weight"], config_dict["charge_weight"], config_dict["dipole_weight"]
    loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, action=config_dict["action"], auto_sol=config_dict["auto_sol"])

    # ------------------- Setting up model and optimizer ------------------ #
    # Normalization of PhysNet atom-wise prediction
    if config_dict["action"] == "E":
        mean_atom, std_atom = atom_mean_std(getattr(data_provider.data, config_dict["action"]),
                                            data_provider.data.N, train_index)
        mean_atom = mean_atom.item()
    elif isinstance(config_dict["action"], list):
        mean_atom = []
        std_atom = []
        for name in config_dict["action"]:
            this_mean, this_std = atom_mean_std(getattr(data_provider.data, name), data_provider.data.N, train_index)
            mean_atom.append(this_mean)
            std_atom.append(this_std)
        mean_atom = torch.as_tensor(mean_atom)
        std_atom = torch.as_tensor(std_atom)
    elif config_dict["action"] == "solubility":
        raise NotImplemented
    else:
        raise ValueError("Invalid action: {}".format(config_dict["action"]))
    E_atomic_scale = std_atom
    E_atomic_shift = mean_atom

    net_kwargs['energy_shift'] = E_atomic_shift
    net_kwargs['energy_scale'] = E_atomic_scale

    net = PhysDimeNet(**net_kwargs)
    shadow_net = PhysDimeNet(**net_kwargs)
    net = net.to(device)
    net = net.type(floating_type)
    shadow_net = shadow_net.to(device)
    shadow_net = shadow_net.type(floating_type)
    shadow_net.load_state_dict(net.state_dict())

    if config_dict["use_swag"]:
        dummy_model = PhysDimeNet(**net_kwargs).to(device).type(floating_type)
        swag_model = SWAG(dummy_model, no_cov_mat=False, max_num_models=20)
    else:
        swag_model = None

    # retrain model
    if config_dict["use_trained_model"]:
        trained_model_dir = glob.glob(config_dict["use_trained_model"])[0]
        logger.info('using trained model: {}'.format(config_dict["use_trained_model"]))

        if os.path.exists(os.path.join(trained_model_dir, 'training_model.pt')):
            net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'training_model.pt'), map_location=device),
                                strict=False)
        else:
            net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pt'), map_location=device),
                                strict=False)
        incompatible_keys = shadow_net.load_state_dict(torch.load(os.path.join(trained_model_dir, 'best_model.pt'),
                                                                  map_location=device), strict=False)
        logger.info("---------incompatible keys-----------")
        logger.info(str(incompatible_keys))
    else:
        trained_model_dir = None

    # model freeze options (transfer learning)
    if config_dict["freeze_option"] == 'prev':
        net.freeze_prev_layers(freeze_extra=False)
    elif config_dict["freeze_option"] == 'prev_extra':
        net.freeze_prev_layers(freeze_extra=True)
    elif config_dict["freeze_option"] == 'none':
        pass
    else:
        raise ValueError('Invalid freeze option: {}'.format(config_dict["freeze_option"]))

    # optimizers
    if config_dict["optimizer"].split('_')[0] == 'emaAms':
        optimizer = EmaAmsGrad(net.parameters(), shadow_net, lr=config_dict["learning_rate"],
                               ema=float(config_dict["optimizer"].split('_')[1]))
    elif config_dict["optimizer"].split('_')[0] == 'sgd':
        optimizer = MySGD(net, lr=config_dict["learning_rate"])
    else:
        raise ValueError('Unrecognized optimizer: {}'.format(config_dict["optimizer"]))
    if config_dict["use_trained_model"] and (not config_dict["reset_optimizer"]):
        if os.path.exists(os.path.join(trained_model_dir, "best_model_optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_optimizer.pt")))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config_dict["decay_steps"], gamma=0.1)
    warm_up_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, total_epoch=config_dict["warm_up_steps"], after_scheduler=scheduler) \
        if config_dict["warm_up_steps"] > 0 else scheduler

    # --------------------- Printing meta data ---------------------- #
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

    # ---------------------- Training ----------------------- #
    logger.info('start training...')

    shadow_net = optimizer.shadow_model
    t0 = time.time()
    val_loss = val_step_new(shadow_net, val_data_loader, loss_fn)

    with open(osp.join(run_directory, "loss_data.csv"), "a") as f:
        f.write("epoch,train_loss,valid_loss,delta_time")
        for key in val_loss.keys():
            if key != "loss":
                f.write(",{}".format(key))
        f.write("\n")

        f.write("0,-1,{},{}".format(val_loss["loss"], time.time()-t0))
        for key in val_loss.keys():
            if key != "loss":
                f.write(",{}".format(val_loss[key]))
        f.write("\n")
        t0 = time.time()
    best_loss = val_loss['loss']

    logger.info('Init lr: {}'.format(get_lr(optimizer)))
    loss_data = []
    early_stop_count = 0
    for epoch in range(config_dict["num_epochs"]):
        train_loss = 0.
        for batch_num, data in enumerate(train_data_loader):
            this_size = data.E.shape[0]

            train_loss += train_step(net, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
                                     max_norm=config_dict["max_norm"],
                                     warm_up_scheduler=warm_up_scheduler) * this_size / train_size

            if config_dict["debug_mode"] & ((batch_num + 1) % 600 == 0):
                logger.info("Batch num: {}/{}, train loss: {} ".format(batch_num, num_train_batches, train_loss))

        logger.info('epoch {} ended, learning rate: {} '.format(epoch, get_lr(optimizer)))
        shadow_net = optimizer.shadow_model
        val_loss = val_step_new(shadow_net, val_data_loader, loss_fn)

        _loss_data_this_epoch = {'epoch': epoch, 't_loss': train_loss, 'v_loss': val_loss['loss'], 'time': time.time()}
        _loss_data_this_epoch.update(val_loss)
        loss_data.append(_loss_data_this_epoch)
        torch.save(loss_data, os.path.join(run_directory, 'loss_data.pt'))
        with open(osp.join(run_directory, "loss_data.csv"), "a") as f:
            f.write("{},{},{},{}".format(epoch, train_loss, val_loss["loss"], time.time()-t0))
            for key in val_loss.keys():
                if key != "loss":
                    f.write(",{}".format(val_loss[key]))
            f.write("\n")
        if config_dict["use_swag"]:
            start, freq = config_dict["uncertainty_modify"].split('_')[1], config_dict["uncertainty_modify"].split('_')[2]
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
            if early_stop_count == config_dict["early_stop"]:
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
    if args.action == "names":
        args.action = args.target_names

    data_provider_class, _kwargs = data_provider_solver(args.data_provider, default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    data_provider = data_provider_class(**_kwargs)
    train(args, data_provider)


if __name__ == "__main__":
    main()
