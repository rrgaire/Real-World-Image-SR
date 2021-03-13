
import os.path
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
        self.paths_LR = None
        self.paths_HR = None

        self.paths_LR = util.get_image_paths(opt['LR_dir'])
        self.paths_HR = util.get_image_paths(opt['HR_dir'])
        print(len(self.paths_LR))

        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, index):
        
        HR_path, LR_path = None, None
        
        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(HR_path)
       
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(LR_path)

        if self.phase == 'train':

            scale = self.opt['scale']
            HR_size = self.opt['HR_patch']

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['flip'], \
                self.opt['rotation'])

       
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)