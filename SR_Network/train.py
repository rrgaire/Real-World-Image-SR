import sys
import torch
import math
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import logging

import utils.utils as util
from utils import data
# from utils.trainer import Trainer as Model
from utils.pre_trainer import Trainer as Model  # for pre-training


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    config_path = sys.argv[1]
    opt = util.load_yaml(config_path)

    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])

    else:
        resume_state = None
        util.mkdir(opt['path']['log'])

    util.setup_logger(None, opt['path']['log'],
                      'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    set_random_seed(0)

    # tensorboard log
    writer = SummaryWriter(log_dir= opt['path']['tb_logger'])

    torch.backends.cudnn.benckmark = True

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = data.create_dataset(dataset_opt, phase)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'valid':
            val_set = data.create_dataset(dataset_opt, phase)
            val_loader = data.create_dataloader(val_set, dataset_opt, phase)
            logger.info('Number of validation images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                             len(val_set)))
        else:
            raise NotImplementedError(
                'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model

    model = Model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.load_model(current_step)
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))

    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            model.train(train_data, current_step)

            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    writer.add_scalar(k, v, current_step)
                logger.info(message)

            if current_step % opt['train']['val_freq'] == 0:
                psnr, ssim = model.validate(val_loader, current_step)

                # log
                logger.info(
                    '# Validation # PSNR: {:.4e} SSIM: {:.4e}'.format(psnr, ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                    epoch, current_step, psnr, ssim))
                # tensorboard logger
                writer.add_scalar('VAL_PSNR', psnr, current_step)
                writer.add_scalar('VAL_SSIM', ssim, current_step)

             # save models and training states
            if current_step % opt['train']['save_step'] == 0:
                logger.info('Saving models and training states.')
                model.save_model(epoch, current_step)


if __name__ == '__main__':
    main()
