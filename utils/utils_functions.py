import gc
import math
import re
import time

import numpy as np
import torch
import torch_geometric

from matplotlib.cbook import deprecated
from scipy.spatial import KDTree
from torch_scatter import scatter

# Constants:

# Coulomb’s constant in eV A and e
from utils.time_meta import record_data

k_e = 14.399645352
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CPU device, part of the functions are calculated faster in cpu
cpu_device = torch.device('cpu')
# Convert Hartree to eV
hartree2ev = 27.2114
# floating type
floating_type = torch.double

kcal2ev = 1 / 23.06035
# Atomic reference energy at 0K (unit: Hartree)
atom_ref = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}

matrix_to_index_map = {}

mae_fn = torch.nn.L1Loss(reduction='mean')
mse_fn = torch.nn.MSELoss(reduction='mean')


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def gaussian_rbf(D, centers, widths, cutoff):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2)
    return rbf


def _get_index_from_matrix(num, previous_num):
    """
    get the edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    if num in matrix_to_index_map.keys():
        return matrix_to_index_map[num] + previous_num
    else:
        index = torch.LongTensor(2, num * num).to(device)
        index[0, :] = torch.cat([torch.zeros(num, device=device).long().fill_(i) for i in range(num)], dim=0)
        index[1, :] = torch.cat([torch.arange(num, device=device).long() for i in range(num)], dim=0)
        mask = (index[0, :] != index[1, :])
        matrix_to_index_map[num] = index[:, mask]
        return matrix_to_index_map[num] + previous_num


matrix_modify = {}


def _get_modify_matrix(num):
    """
    get the modify matrix.
    equivalent to -torch.eye(num)
    data will be stored in matrix_modify to save time when next time need it
    :param num:
    :return:
    """
    if num in matrix_modify.keys():
        return matrix_modify[num]
    else:
        matrix = torch.Tensor(num, num).type(floating_type).zero_()
        for i in range(num):
            matrix[i, i] = -1.
        matrix_modify[num] = matrix
        return matrix


batch_pattern = {}


def _get_batch_pattern(batch_size, max_num):
    """
    get the batch pattern, for example, if batch_size=5, max_num=3
    the pattern will be: [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    new pattern will be stored in batch_pattern dictionary to avoid recalculation
    :return:
    """
    if batch_size in batch_pattern.keys():
        return batch_pattern[batch_size]
    else:
        pattern = [i // max_num for i in range(batch_size * max_num)]
        batch_pattern[batch_size] = pattern
        return pattern


def _cal_dist(d1, d2):
    """
    calculate the Euclidean distance between d1 and d2
    :param d1:
    :param d2:
    :return:
    """
    delta_R = d1 - d2
    return torch.sqrt(torch.sum(torch.mul(delta_R, delta_R))).view(-1, 1).type(floating_type)


def softplus_inverse(x):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    return torch.log(-torch.expm1(-x)) + x


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def _chi_ij(R_ij, cutoff):
    """
    Chi(Rij) function which is used to calculate long-range energy
    return 0 when R_ij = -1 (use -1 instead of 0 to prevent nan when backward)
    :return: Chi(Rij)
    """
    return torch.where(R_ij != -1, _cutoff_fn(2 * R_ij, cutoff) / torch.sqrt(torch.mul(R_ij, R_ij) + 1) +
                       (1 - _cutoff_fn(2 * R_ij, cutoff)) / R_ij, torch.zeros_like(R_ij))


def _correct_q(qi, N, atom_to_mol_batch, q_ref):
    """
    calculate corrected partial_q in PhysNet
    :param qi: partial charge predicted by PhysNet, shape(-1, 1)
    :return: corrected partial_q, shape(-1, 1)
    """
    # predicted sum charge
    Q_pred = scatter(reduce='add', src=qi, index=atom_to_mol_batch, dim=0)

    correct_term = (Q_pred - q_ref) / (N.type(floating_type).to(device))
    # broad cast according to batch to make sure dim correct
    broadcasted_correct_term = correct_term.take(atom_to_mol_batch)
    return qi - broadcasted_correct_term


def cal_coulomb_E(qi: torch.Tensor, edge_dist, edge_index, cutoff, q_ref, N, atom_mol_batch):
    """
    Calculate coulomb Energy from chi(Rij) and corrected q
    Calculate ATOM-WISE energy!
    :return: calculated E
    """
    # debug: cal passed time to improve code efficiency
    # print('&' * 40)
    # t0 = time.time()

    cutoff = cutoff.to(device)

    # debug: cal passed time to improve code
    # print('T--------pre cal: ', time.time() - t0)
    if q_ref is not None:
        """
        This correction term is from PhysNet paper:
        As neural networks are a purely numerical algorithm, it is not guaranteed a priori
        that the sum of all predicted atomic partial charges qi is equal
        to the total charge Q (although the result is usually very close
        when the neural network is properly trained), so a correction
        scheme like eq 14 is necessary to guarantee charge
        conservation
        """
        assert N is not None
        assert atom_mol_batch is not None
        qi = _correct_q(qi, N, atom_mol_batch, q_ref)

    # Qi will be corrected according to the paper
    # qi = _partial_q(qi, N, atom_to_mol_batch, Q)

    # debug: cal passed time to improve code
    # print('T--------cal partial qi: ', time.time() - t0)

    # debug: cal passed time to improve code
    # print('T--------split: ', time.time() - t0)

    q_first = qi.take(edge_index[0, :]).view(-1, 1)
    q_second = qi.take(edge_index[1, :]).view(-1, 1)
    revised_dist = _chi_ij(edge_dist, cutoff=cutoff)
    coulomb_E_terms = q_first * revised_dist * q_second
    '''
    set dim_size here in case the last batch has only one 'atom', which will cause dim to be 1 less because no 
    edge will be formed th that way 
    '''
    coulomb_E = scatter(reduce='add', src=coulomb_E_terms.view(-1), index=edge_index[0, :], dim_size=qi.shape[0], dim=0)

    # debug: cal passed time to improve code
    # print('T--------for loop: ', time.time() - t0)
    # print('&' * 40)

    # times 1/2 because of the double counting
    return (coulomb_E / 2).to(device)


def cal_p(qi, R, atom_to_mol_batch):
    """
    Calculate pi from qi and molecule coordinate
    :return: pi
    """

    tmp = torch.mul(qi.view(-1, 1), R.to(device))
    p = scatter(reduce='add', src=tmp, index=atom_to_mol_batch.to(device), dim=0)
    return p


def cal_edge(R, N, prev_N, edge_index, cal_coulomb=True):
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
    :param cal_coulomb:
    :param prev_N:
    :param edge_index:
    :param R:
    :param N:
    :return:
    """
    if cal_coulomb:
        '''
        IMPORTANT: DO NOT use num(tensor) itself as input, which will be regarded as dictionary key in this function,
        use int value(num.item())
        Using tensor as dictionary key will cause unexpected problem, for example, memory leak
        '''
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), previous_num) for num, previous_num in zip(N, prev_N)], dim=-1)
        points1 = R[coulomb_index[0, :], :]
        points2 = R[coulomb_index[1, :], :]
        coulomb_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        coulomb_dist = torch.sqrt(coulomb_dist)

    else:
        coulomb_dist = None
        coulomb_index = None

    short_range_index = edge_index
    points1 = R[edge_index[0, :], :]
    points2 = R[edge_index[1, :], :]
    short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
    short_range_dist = torch.sqrt(short_range_dist)
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def get_batch(atom_map, max_num):
    """
    from map to batch
    :param atom_map:
    :param atom_map, batch_size, max_num:
    :return: batch, example: [0,0,0,0,0,0,1,1,2,2,2,...]
    """
    batch_size = atom_map.shape[0] // max_num
    pattern = _get_batch_pattern(batch_size, max_num)
    return torch.LongTensor(pattern)[atom_map]


def get_uniform_variance(n1, n2):
    """
    get the uniform variance to initialize the weight of DNNs suggested at
    Glorot,X.;Bengio,Y. Understanding the Difficulty of Training Deep Feed forward Neural Networks.
    Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010; pp 249–256.
    :param n1: the size of previous layer
    :param n2: the size of next layer :return: uniform variance
    """
    return math.sqrt(6) / math.sqrt(n1 + n2)


# generates a random square orthogonal matrix of dimension dim
def square_orthogonal_matrix(dim=3, seed=None):
    random_state = np.random
    if seed is not None:  # allows to get the same matrix every time
        random_state.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# generates a random (semi-)orthogonal matrix of size NxM
def semi_orthogonal_matrix(N, M, seed=None):
    if N > M:  # number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix(dim=N, seed=seed)
    else:  # number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix(dim=M, seed=seed)
    return square_matrix[:N, :M]


# generates a weight matrix with variance according to Glorot initialization
# based on a random (semi-)orthogonal matrix
# neural networks are expected to learn better when features are decorrelated
# (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
# "Dropout: a simple way to prevent neural networks from overfitting",
# "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
def semi_orthogonal_glorot_weights(n_in, n_out, scale=2.0, seed=None):
    W = semi_orthogonal_matrix(n_in, n_out, seed=seed)
    W *= np.sqrt(scale / ((n_in + n_out) * W.var()))
    return torch.Tensor(W).type(floating_type).t()


def get_atom_to_efgs_batch(efgs_batch, num_efgs, atom_to_mol_mask):
    # efgs_batch: shape(n_batch, 29)
    # num_efgs: shape(n_batch)

    _batch_corrector = torch.zeros_like(num_efgs)
    _batch_corrector[1:] = num_efgs[:-1]
    batch_size = _batch_corrector.shape[0]
    for i in range(1, batch_size):
        _batch_corrector[i] = _batch_corrector[i - 1] + _batch_corrector[i]
    _batch_corrector = _batch_corrector.view(-1, 1)  # make sure correct dimension for broadcasting
    efgs_batch = efgs_batch + _batch_corrector
    atom_to_efgs_batch = efgs_batch.view(-1)[atom_to_mol_mask]
    return atom_to_efgs_batch


def get_kd_tree_array(R, N):
    """
    Used in data_provider, encapsulate coordinates and numbers into a kd_tree array(tensor)
    :param R: Coordinates
    :param N: Number of atoms in this molecule
    :return: tensor of KD_Tree instances
    """
    num_molecules = R.shape[0]
    kd_trees = np.empty(num_molecules, dtype=KDTree)
    for i in range(num_molecules):
        kd_trees[i] = KDTree(R[i, :N[i].item(), :])
    return kd_trees


def atom_mean_std(E, N, index):
    """
    calculate the mean and stand variance of Energy in the training set
    :return:
    """
    mean = 0.0
    std = 0.0
    num = len(index)
    for _i in range(num):
        i = index[_i]
        m_prev = mean
        x = E[i] / N[i]
        mean += (x - mean) / (i + 1)
        std += (x - mean) * (x - m_prev)
    std = math.sqrt(std / num)
    return mean, std


def _pre_nums(N, i):
    return N[i - 1] if i > 0 else 0


def load_data_from_index(dataset, indexes):
    """
    Similar to Batch.to_data_list() method in torch_geometric
    :param indexes:
    :param dataset:
    :return:
    """
    result = collate_fn([dataset[i.item()] for i in indexes], clone=False)
    return result


def _cal_dim(key):
    return -1 if re.search("index", key) else 0


def collate_fn(data_list, clone=False):
    """
    Note: using clone here, maybe not efficient
    :param clone:
    :param data_list:
    :return:
    """
    # torch.cuda.synchronize(device)
    # t0 = time.time()

    if clone:
        data_list = [data.clone() for data in data_list]
    batch = torch_geometric.data.Data()
    batch_size = len(data_list)
    keys = data_list[0].keys
    for key in keys:
        batch[key] = torch.cat([data_list[i][key] for i in range(batch_size)], dim=_cal_dim(key))

    cum = {}
    for key in keys:
        if re.search('num_', key) or (key == 'N'):
            cum_sum_before = torch.zeros_like(batch[key])
            cum_sum_before[1:] = torch.cumsum(batch[key], dim=0)[:-1]
            # key[4:] means:   num_B_edge -> B_edge
            cum_name = 'N' if key == 'N' else key[4:]
            cum[cum_name] = cum_sum_before
    for key in keys:
        if re.search('edge_index', key):
            # BN_edge_index -> BN_edge
            # B_msg_edge_index -> B_msg_edge
            edge_name = key[:-6]
            if re.search('_msg_edge_index', key):
                # B_msg_edge_index -> B
                bond_type = key[:-15]
                batch[key + '_correct'] = torch.repeat_interleave(cum[bond_type + '_edge'], batch['num_' + edge_name])
            else:
                batch[key + '_correct'] = torch.repeat_interleave(cum['N'], batch['num_' + edge_name])
    batch.atom_mol_batch = torch.repeat_interleave(torch.arange(batch['N'].shape[0]), batch['N'])

    # t0 = record_data('collate fn', t0)
    return batch.contiguous().to(device)


def dime_edge_expansion(R, edge_index, msg_edge_index, n_dime_rbf, dist_calculator, bessel_calculator,
                        feature_interact_dist, cos_theta=True, **kwargs):
    # t0 = record_data('edge_msg_gen.load_data', t0)

    """
    calculating bonding infos
    those data will be used in DimeNet modules
    """
    dist_atom = dist_calculator(R[edge_index[0, :], :], R[edge_index[1, :], :])
    rbf_ji = bessel_calculator.cal_rbf(dist_atom, feature_interact_dist, n_dime_rbf)

    # t0 = record_data('edge_msg_gen.bond_rbf', t0)

    dist_msg = dist_calculator(
        R[edge_index[0, msg_edge_index[1, :]], :], R[edge_index[1, msg_edge_index[1, :]], :]
    ).view(-1, 1)
    angle_msg = cal_angle(
        R, edge_index[:, msg_edge_index[0, :]], edge_index[:, msg_edge_index[1, :]], cos_theta).view(-1, 1)
    sbf_kji = bessel_calculator.cal_sbf(dist_msg, angle_msg, feature_interact_dist)

    # t0 = record_data('edge_msg_gen.bond_sbf', t0)
    return rbf_ji, sbf_kji


@deprecated('0.0.1')
def phys_edge_expansion(R, edge_index, n_phys_rbf, dist_calculator, bessel_calculator, feature_interact_dist):
    dist_atom_non_bond = dist_calculator(R[edge_index[0, :], :], R[edge_index[1, :], :])
    rbf = bessel_calculator.cal_rbf(dist_atom_non_bond, feature_interact_dist, n_phys_rbf)

    return rbf


def get_n_params(model, logger=None):
    """
    Calculate num of parameters in the model
    :param logger:
    :param model:
    :return:
    """
    result = ''
    for name, param in model.named_parameters():
        if logger is not None:
            logger.info('{}: {}'.format(name, param.data.shape))
        result = result + '{}: {}\n'.format(name, param.data.shape)
    return sum([x.nelement() for x in model.parameters()]), result


def cal_angle(R, edge1, edge2, cal_cos_theta):
    delta_R1 = R[edge1[0, :], :] - R[edge1[1, :], :]
    delta_R2 = R[edge2[0, :], :] - R[edge2[1, :], :]
    inner = torch.sum(delta_R1 * delta_R2, dim=-1)
    delta_R1_l = torch.sqrt(torch.sum(delta_R1 ** 2, dim=-1))
    delta_R2_l = torch.sqrt(torch.sum(delta_R2 ** 2, dim=-1))
    cos_theta = inner / (delta_R1_l * delta_R2_l + 1e-7)
    if cal_cos_theta:
        angle = cos_theta
    else:
        angle = torch.acos(cos_theta)
    return angle.view(-1, 1)


def get_tensors():
    """
    print out tensors in current system to debug memory leak
    :return: set of infos about tensors
    """
    result = {"set_init"}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tup = (obj.__hash__(), obj.size())
                result.add(tup)
        except:
            pass
    print('*' * 30)
    return result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def info_resolver(s):
    """
    Internal function which resolve expansion function into details, eg:
    gaussian_64_10.0 means gaussian expansion, n=64 and cutoff=10.0
    :param s:
    :return:
    """
    info = s.split('_')
    result = {'name': info[0]}
    if info[0] == 'bessel' or info[0] == 'gaussian':
        result['n'] = int(info[1])
        result['dist'] = float(info[2])
    elif info[0] == 'defaultDime':
        result['n'] = int(info[1])
        result['envelop_p'] = int(info[2])
        result['n_srbf'] = int(info[3])
        result['n_shbf'] = int(info[4])
        result['dist'] = float(info[5])
    elif info[0] == 'coulomb':
        result['dist'] = float(info[1])
    elif info[0] == 'none':
        pass
    else:
        raise ValueError(f"Invalid expansion function {s} !")
    return result


def expansion_splitter(s):
    """
    Internal use only
    Strip expansion function into a dictionary
    :param s:
    :return:
    """
    result = {}
    for mapping in s.split(' '):
        value = mapping.split(':')[1]
        keys = mapping.split(':')[0]
        if keys[0] == '(':
            assert keys[-1] == ')'
            keys = keys[1:-1]
            for key in keys.split(','):
                result[key.strip()] = value
        else:
            result[keys.strip()] = value
    return result


def error_message(value, name):
    raise ValueError('Invalid {} : {}'.format(name, value))


def print_val_results(dataset_name, loss, emae, ermse, qmae, qrmse, pmae, prmse):
    log_info = 'Validating {}: '.format(dataset_name)
    log_info += (' loss: {:.6f} '.format(loss))
    log_info += ('emae: {:.6f} '.format(emae))
    log_info += ('ermse: {:.6f} '.format(ermse))
    log_info += ('qmae: {:.6f} '.format(qmae))
    log_info += ('qrmse: {:.6f} '.format(qrmse))
    log_info += ('pmae: {:.6f} '.format(pmae))
    log_info += ('prmse: {:.6f} '.format(prmse))
    return log_info


def option_solver(option_txt):
    if len(option_txt.split('[')) == 1:
        return {}
    else:
        # option_txt should be like :    '[n_read_out=2,other_option=value]'
        # which will be converted into a dictionary: {n_read_out: 2, other_option: value}
        option_txt = option_txt.split('[')[1]
        option_txt = option_txt[:-1]
        return {argument.split('=')[0]: argument.split('=')[1]
                for argument in option_txt.split(',')}


def kwargs_solver(args):
    debug_mode = (args.debug_mode.lower() == 'true')
    normalize = (args.normalize.lower() == 'true')
    shared_normalize_param = (args.shared_normalize_param.lower() == 'true')
    restrain_non_bond_pred = (args.restrain_non_bond_pred.lower() == 'true')
    coulomb_charge_correct = (args.coulomb_charge_correct.lower() == 'true')

    DNNKwargs = {'n_atom_embedding': 95,
                 'n_feature': args.n_feature,
                 'n_output': args.n_output,
                 'n_dime_before_residual': args.n_dime_before_residual,
                 'n_dime_after_residual': args.n_dime_after_residual,
                 'n_output_dense': args.n_output_dense,
                 'n_phys_atomic_res': args.n_phys_atomic_res,
                 'n_phys_interaction_res': args.n_phys_interaction_res,
                 'n_phys_output_res': args.n_phys_output_res,
                 'n_bi_linear': args.n_bi_linear,
                 'nh_lambda': args.nh_lambda,
                 'normalize': normalize,
                 'debug_mode': debug_mode,
                 'activations': args.activations,
                 'shared_normalize_param': shared_normalize_param,
                 'restrain_non_bond_pred': restrain_non_bond_pred,
                 'expansion_fn': args.expansion_fn,
                 'modules': args.modules,
                 'bonding_type': args.bonding_type,
                 'uncertainty_modify': args.uncertainty_modify,
                 'coulomb_charge_correct': coulomb_charge_correct,
                 'action': args.action
                 }
    return DNNKwargs


def add_parser_arguments(parser):
    """
    add arguments to parser
    :param parser:
    :return: added parser
    """
    parser.add_argument('--debug_mode', type=str)
    parser.add_argument('--modules', type=str, help="eg: D P D P D P, D for DimeNet and P for PhysNet")
    parser.add_argument('--bonding_type', type=str, help="eg: B N B N B N, B for bonding-edge, N for non-bonding "
                                                         "edge. (future) L for long-range interaction and BN for both "
                                                         "bonding and non-bonding")
    parser.add_argument('--n_feature', type=int)
    parser.add_argument('--n_dime_before_residual', type=int)
    parser.add_argument('--n_dime_after_residual', type=int)
    parser.add_argument('--n_output_dense', type=int)
    parser.add_argument('--n_phys_atomic_res', type=int)
    parser.add_argument('--n_phys_interaction_res', type=int)
    parser.add_argument('--n_phys_output_res', type=int)
    parser.add_argument('--n_bi_linear', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--ema_decay', type=float, help='Deprecated, use --optimizer option instead')
    parser.add_argument('--l2lambda', type=float)
    parser.add_argument('--nh_lambda', type=float)
    parser.add_argument('--decay_steps', type=int)
    parser.add_argument('--decay_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--valid_batch_size', type=int)
    parser.add_argument('--force_weight', type=float)
    parser.add_argument('--charge_weight', type=float)
    parser.add_argument('--dipole_weight', type=float)
    parser.add_argument('--use_trained_model', type=str)
    parser.add_argument('--max_norm', type=float)
    parser.add_argument('--log_file_name', type=str)
    parser.add_argument('--folder_prefix', type=str)
    parser.add_argument('--config_name', type=str, default='config.txt')
    parser.add_argument('--normalize', type=str)
    parser.add_argument('--shared_normalize_param', type=str)
    parser.add_argument('--edge_version', type=str, help="voronoi | cutoff")
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--boundary_factor', type=float)
    parser.add_argument('--activations', type=str, help='swish | ssp')
    parser.add_argument('--expansion_fn', type=str)
    parser.add_argument('--restrain_non_bond_pred', type=str)
    parser.add_argument('--frag9_train_size', type=int, help="solely used for training curve")
    parser.add_argument('--frag20_train_size', type=int, help="solely used for training curve")
    parser.add_argument('--test_interval', type=str, help="DONT USE! For compatibility only, no longer used.")
    parser.add_argument('--warm_up_steps', type=int, help="Steps to warm up")
    parser.add_argument('--data_provider', type=str, help="Data provider arguments:"
                                                          " qm9 | frag9to20_jianing | frag9to20_all")
    parser.add_argument('--uncertainty_modify', type=str, default='none',
                        help="none | concreteDropoutModule | concreteDropoutOutput | swag_${start}_${freq}")
    parser.add_argument('--early_stop', type=int, default=-1, help="early stopping, set to -1 to disable")
    parser.add_argument('--optimizer', type=str, default='emaAms_0.999', help="emaAms_${ema} | sgd")
    parser.add_argument('--freeze_option', type=str, default='none', help='none | prev | prev_extra')
    parser.add_argument('--comment', type=str, help='just comment')
    parser.add_argument('--remove_atom_ids', type=int, default=5, help='remove atoms from dataset')
    parser.add_argument('--coulomb_charge_correct', type=str, default="False",
                        help='calculate charge correction when calculation Coulomb interaction')
    parser.add_argument('--reset_optimizer', type=str, default="True",
                        help='If true, will reset optimizer regardless of if you use pretrained model or not')
    parser.add_argument("--n_output", type=int, default=2, help="number of outputs, defaults to 2 for energy and charge"
                                                                "predictions.")
    parser.add_argument("--action", type=str, default="E", help="name of target, must be consistent with name in"
                                                                "data_provider, default E is for PhysNet energy")
    parser.add_argument("--target_names", type=str, action="append", default=[])
    return parser


def remove_handler(log):
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    return


if __name__ == '__main__':
    map = [True] * 32 * 6
    map[3] = False
    print(get_batch(map, 32, 6))