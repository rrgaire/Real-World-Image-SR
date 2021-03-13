import random
import numpy as np
import torch

import utils.utils as util


def create_dataset(dataset_opt, phase):

    dataset = Dataset(dataset_opt, phase)

    return dataset


def create_dataloader(dataset, dataset_opt, phase):

    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, phase):
        super(Dataset, self).__init__()
        self.opt = opt
        self.phase = phase
        self.paths_REAL = None
        self.paths_BIC = None

        self.paths_REAL = util.get_image_paths(opt['REAL_dir'])
        self.paths_BIC = util.get_image_paths(opt['BIC_dir'])
        print(len(self.paths_REAL))

        if self.paths_REAL and self.paths_BIC:
            assert len(self.paths_REAL) == len(self.paths_BIC), \
                'BIC_LR and REAL_LR datasets have different number of images - {}, {}.'.format(
                len(self.paths_REAL), len(self.paths_BIC))

    def __getitem__(self, index):

        BIC_path, REAL_path = None, None

        # get HR image
        BIC_path = self.paths_BIC[index]
        img_BIC = util.read_img(BIC_path)

        REAL_path = self.paths_REAL[index]
        img_REAL = util.read_img(REAL_path)

        if self.phase == 'train':

            patch_size = self.opt['patch']

            H, W, C = img_REAL.shape

            # randomly crop
            rnd_h = random.randint(0, max(0, H - patch_size))
            rnd_w = random.randint(0, max(0, W - patch_size))

            img_REAL = img_REAL[rnd_h:rnd_h +
                                patch_size, rnd_w:rnd_w + patch_size, :]
            img_BIC = img_BIC[rnd_h:rnd_h + patch_size,
                              rnd_w:rnd_w + patch_size, :]

            # image augmentation - flip, rotate
            img_REAL, img_BIC = util.augment([img_REAL, img_BIC], self.opt['flip'],
                                             self.opt['rotation'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_BIC.shape[2] == 3:
            img_BIC = img_BIC[:, :, [2, 1, 0]]
            img_REAL = img_REAL[:, :, [2, 1, 0]]
        img_BIC = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_BIC, (2, 0, 1)))).float()
        img_REAL = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_REAL, (2, 0, 1)))).float()

        return {'REAL': img_REAL, 'BIC': img_BIC, 'REAL_path': REAL_path, 'BIC_path': BIC_path}

    def __len__(self):
        return len(self.paths_BIC)
