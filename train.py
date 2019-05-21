import os
import torch
import numpy as np
import sys

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_network

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def main():
    # define output folder
    output_dir = './saved_models/'
    log_dir = './mae_mse/'
    checkpoint_dir = './checkpoint/'

    train_path = '/home/khuong/Desktop/Projects/python/dataset/crowd_counting/SH_B/train/images'
    train_gt_path = '/home/khuong/Desktop/Projects/python/dataset/crowd_counting/SH_B/train/ground_truth'
    val_path = '/home/khuong/Desktop/Projects/python/dataset/crowd_counting/SH_B/val/images'
    val_gt_path = '/home/khuong/Desktop/Projects/python/dataset/crowd_counting/SH_B/val/ground_truth'

    # last checkpoint
    checkpointfile = os.path.join(checkpoint_dir, 'checkpoint.74.pth.tar')

    # some description
    method = 'mcnn'
    dataset_name = 'SH_B'

    # log file
    f_train_loss = open(os.path.join(log_dir, "train_loss.csv"), "a+")
    f_val_loss = open(os.path.join(log_dir, "val_loss.csv"), "a+")

    # Training configuration
    start_epoch = 0
    end_epoch = 2000
    lr = 0.00001
    # momentum = 0.9
    disp_interval = 1000
    # log_interval = 250

    # Flag
    CONTINUE_TRAIN = True
    # Tensorboard  config

    # use_tensorboard = False
    # save_exp_name = method + '_' + dataset_name + '_' + 'v1'
    # remove_all_log = False   # remove all historical experiments in TensorBoard
    # exp_name = None # the previous experiment name in TensorBoard

    # -----------------------------------------------------------------------------------------
    rand_seed = 64678
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

    # Define network
    net = CrowdCounter()
    network.weights_normal_init(net, dev=0.01)
    # net.cuda()
    net.train()
    # params = list(net.parameters())
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # # tensorboad
    # use_tensorboard = use_tensorboard and CrayonClient is not None
    # if use_tensorboard:
    #     cc = CrayonClient(hostname='127.0.0.1')
    #     if remove_all_log:
    #         cc.remove_all_experiments()
    #     if exp_name is None:
    #         exp_name = save_exp_name
    #         exp = cc.create_experiment(exp_name)
    #     else:
    #         exp = cc.open_experiment(exp_name)

    # training param

    if CONTINUE_TRAIN:
        net, optimizer, start_epoch = utils.load_checkpoint(
            net, optimizer, filename=checkpointfile)

    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()

    # Load data
    data_loader = ImageDataLoader(
        train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(
        val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
    best_mae = sys.maxsize

    # Start training

    for this_epoch in range(start_epoch, end_epoch-1):
        step = -1
        train_loss = 0
        for blob in data_loader:
            step += 1
            img_data = blob['data']
            gt_data = blob['gt_density']
            et_data = net(img_data, gt_data)
            loss = net.loss
            train_loss += loss.data
            step_cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration
                gt_count = np.sum(gt_data)
                et_data = et_data.data.cpu().numpy()
                et_count = np.sum(et_data)
                utils.save_results(img_data, gt_data, et_data, output_dir,
                                   fname="{}.{}.png".format(this_epoch, step))
                log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (this_epoch,
                                                                                                step, 1./fps, gt_count, et_count)
                log_print(log_text, color='green', attrs=['bold'])
                re_cnt = True

            if re_cnt:
                t.tic()
                re_cnt = False

        # Save checkpoint
        state = {'epoch': this_epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict()}
        cp_filename = "checkpoint.{}.pth.tar".format(this_epoch)
        torch.save(state, os.path.join(checkpoint_dir, cp_filename))
# ========================== END 1 EPOCH==================================================================================
        train_mae, train_mse = evaluate_network(net, data_loader)
        f_train_loss.write("{},{}\n".format(train_mae, train_mse))
        log_text = 'TRAINING - EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (
            this_epoch, train_mae, train_mse)
        log_print(log_text, color='green', attrs=['bold'])
# =====================================================VALIDATION=========================================================
        # calculate error on the validation dataset
        val_mae, val_mse = evaluate_network(net, data_loader_val)
        f_val_loss.write("{},{}\n".format(val_mae, val_mse))
        log_text = 'VALIDATION - EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (
            this_epoch, val_mae, val_mse)
        log_print(log_text, color='green', attrs=['bold'])
        # SAVE model
        is_save = False
        if val_mae <= best_mae:
            if val_mae < best_mae:
                is_save = True
                best_mae = val_mae
                best_mse = val_mse
            else:
                if val_mse < best_mse:
                    is_save = True
                    best_mse = val_mse

        if is_save:
            save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(
                method, dataset_name, this_epoch))
            network.save_net(save_name, net)
            best_model = '{}_{}_{}.h5'.format(method, dataset_name, this_epoch)
            log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (
                best_mae, best_mse, best_model)
            log_print(log_text, color='green', attrs=['bold'])

        # if use_tensorboard:
        #     exp.add_scalar_value('MAE', mae, step=epoch)
        #     exp.add_scalar_value('MSE', mse, step=epoch)
        #     exp.add_scalar_value('train_loss', train_loss /
        #                          data_loader.get_num_samples(), step=epoch)

    f_train_loss.close()
    f_val_loss.close()


if __name__ == "__main__":
    main()
