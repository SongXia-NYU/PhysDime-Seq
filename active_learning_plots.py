import glob
import os
import os.path as osp
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def al_plot_folder(al_folder):
    cycle_folders = glob.glob(osp.join(al_folder, 'cycle*_run_*'))
    for ele in cycle_folders:
        if ele.find('_new_mol_') > 0:
            cycle_folders.remove(ele)
    cycle_folders.sort(key=lambda s: int(osp.basename(s).split('_')[0][5:]))
    cycle_losses = [torch.load(osp.join(cycle_folder, 'loss_data.pt')) for cycle_folder in cycle_folders]
    losses = []
    for cycle_loss in cycle_losses:
        losses.extend([v['v_emae'] for v in cycle_loss])
    losses = np.asarray(losses) * 23.01
    plt.figure(figsize=(15, 10))
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss, kcal/mol')
    plt.title('losses vs epoch')
    plt.savefig(osp.join(al_folder, 'loss_epoch'))
    return


def al_plot_test_loss(test_dir):
    folders = glob.glob(osp.join(test_dir, 'cycle*_test_*'))
    cycle_diff = [torch.load(osp.join(folder, 'diff.pt')) for folder in folders]

    percent_large_1 = [1.*(diff > 1.).sum() / diff.shape[0] for diff in cycle_diff]
    percent_large_10 = [1.*(diff > 10.).sum() / diff.shape[0] for diff in cycle_diff]
    largest_error = [diff.abs().max() for diff in cycle_diff]
    e_mae = [diff.abs().mean() for diff in cycle_diff]
    x = [int(osp.basename(folder).split('_')[0][5:]) for folder in folders]

    for y, y_label, f_name in zip([percent_large_1, percent_large_10, largest_error, e_mae],
                                  ['% larger than 1kcal/mol', '% larger than 10 kcal/mol', 'largest error kcal/mol',
                                   'E_MAE'],
                                  ['larger1', 'larger10', 'largest', 'e_mae']):
        plt.figure(figsize=(15, 10))
        y = [d.item() for d in y]
        plt.bar(x, y, width=0.3)
        plt.xlabel('cycle')
        plt.ylabel(y_label)
        plt.savefig(osp.join(test_dir, f_name))


def al_plot_main():
    folders = glob.glob('exp*_active_*')
    for folder in folders:
        al_plot_folder(folder)


if __name__ == '__main__':
    al_plot_test_loss('../../shared_folder/raw_data/PhysDime/exp163_active_2020-08-20_201505')
