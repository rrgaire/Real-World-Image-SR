import os
import torch
import cv2

import torch.nn as nn
from torch.optim import lr_scheduler


import numpy as np
from collections import OrderedDict

from networks.generator import Generator
from networks.discriminator import Discriminator
from networks.feature_extractor import VGGFeatureExtractor

from utils.losses import cri_pix
from utils.metrics import PSNR, SSIM

import utils.utils as util

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


class Trainer():
    def __init__(self, opt):
        self.device = torch.device('cuda')
        self.opt = opt
        self.G = Generator(self.opt['network_G']).to(self.device)
        util.init_weights(self.G, init_type='kaiming', scale=0.1)
        self.G.train()

        self.log_dict = OrderedDict()

        self.optim_params = [
            v for k, v in self.G.named_parameters() if v.requires_grad]
        self.opt_G = torch.optim.Adam(self.optim_params, lr=self.opt['train']['lr_G'], betas=(
            self.opt['train']['b1_G'], self.opt['train']['b2_G']))

        self.optimizers = [self.opt_G]
        self.schedulers = [lr_scheduler.MultiStepLR(
            optimizer, self.opt['train']['lr_steps'], self.opt['train']['lr_gamma']) for optimizer in self.optimizers]

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_log(self):
        return self.log_dict

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def load_model(self, step, strict=True):
        self.G.load_state_dict(torch.load(
            f"{self.opt['path']['checkpoints']['models']}/{step}_G.pth"), strict=strict)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''

        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def save_network(self, network, network_label, iter_step):

        util.mkdir(self.opt['path']['checkpoints']['models'])
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(
            self.opt['path']['checkpoints']['models'], save_filename)

        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_model(self, epoch, current_step):
        self.save_network(self.G, 'G', current_step)
        self.save_training_state(epoch, current_step)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step,
                 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        util.mkdir(self.opt['path']['checkpoints']['states'])
        save_path = os.path.join(
            self.opt['path']['checkpoints']['states'], save_filename)
        torch.save(state, save_path)

    def train(self, train_batch, step):

        self.reallr = train_batch['REAL'].to(self.device)
        self.biclr = train_batch['BIC'].to(self.device)

        self.opt_G.zero_grad()

        self.fake_bic = self.G(self.reallr)

        l_g_total = 0
        # pixel loss
        l_g_pix = self.opt['train']['wt_pix'] * \
            cri_pix(self.fake_bic, self.biclr)
        l_g_total += l_g_pix

        l_g_total.backward()
        self.opt_G.step()

        # set log
        # G
        self.log_dict['l_g_pix'] = l_g_pix.item()

    def validate(self, val_batch, current_step):
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        for _, val_data in enumerate(val_batch):
            idx += 1
            img_name = os.path.splitext(
                os.path.basename(val_data['REAL_path'][0]))[0]
            img_dir = os.path.join(
                self.opt['path']['checkpoints']['val_image_dir'], img_name)
            util.mkdir(img_dir)

            self.val_REAL = val_data['REAL'].to(self.device)
            self.val_BIC = val_data['BIC'].to(self.device)

            self.G.eval()
            with torch.no_grad():
                self.val_fake_BIC = self.G(self.val_REAL)
            self.G.train()

            val_REAL = self.val_REAL.detach()[0].float().cpu()
            val_BIC = self.val_BIC.detach()[0].float().cpu()
            val_fake_BIC = self.val_fake_BIC.detach()[0].float().cpu()

            fake_bic_img = util.tensor2img(val_fake_BIC)  # uint8
            gt_img = util.tensor2img(val_BIC)  # uint8

            # Save fake_bic images for reference
            save_img_path = os.path.join(img_dir, '{:s}_pre_{:d}.png'.format(
                img_name, current_step))
            cv2.imwrite(save_img_path, fake_bic_img)

            # calculate PSNR
            crop_size = 4
            gt_img = gt_img / 255.
            fake_bic_img = fake_bic_img / 255.
            cropped_fake_bic_img = fake_bic_img[crop_size:-
                                                crop_size, crop_size:-crop_size, :]
            cropped_gt_img = gt_img[crop_size:-
                                    crop_size, crop_size:-crop_size, :]
            avg_psnr += PSNR(cropped_fake_bic_img * 255, cropped_gt_img * 255)
            avg_ssim += SSIM(cropped_fake_bic_img * 255, cropped_gt_img * 255)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        return avg_psnr, avg_ssim
