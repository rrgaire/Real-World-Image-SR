import sys
import os.path
import glob
import cv2
import numpy as np
import torch
from R2B_Network.networks.generator import Generator as r2b_gen
from SR_Network.networks.generator import Generator as sr_gen
import utils.utils as util


config_path = sys.argv[1]
cfg_test = util.load_yaml(config_path)

r2b_model_path = cfg_test['r2b_model_path']
sr_model_path = cfg_test['sr_model_path']

device = torch.device('cuda')
# device = torch.device('cpu')

input_dir = cfg_test['LR_dir']
output_dir = cfg_test['SR_dir']
util.mkdir(output_dir)

r2b_model = r2b_gen(cfg_test['network_R2B'])
sr_model = sr_gen(cfg_test['network_SR'])

r2b_model.load_state_dict(torch.load(r2b_model_path), strict=False)
print('Loaded R2B model at {}'.format(r2b_model_path))
sr_model.load_state_dict(torch.load(sr_model_path), strict=False)
print('Loaded SR model at {}'.format(sr_model_path))

r2b_model.eval()
sr_model.eval()


for k, v in r2b_model.named_parameters():
    v.requires_grad = False
r2b_model = r2b_model.to(device)

for k, v in sr_model.named_parameters():
    v.requires_grad = False
sr_model = sr_model.to(device)

print('Starting Testing...')

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img_inp = cv2.imread(path, cv2.IMREAD_COLOR)

    # R2B model
    img_inp = img_inp * 1.0 / 255
    img_inp = torch.from_numpy(np.transpose(
        img_inp[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_inp = img_inp.unsqueeze(0)
    img_inp = img_inp.to(device)

    img_bic = r2b_model(img_inp).data.squeeze(
    ).float().cpu().clamp_(0, 1).numpy()
    img_bic = np.transpose(img_bic[[2, 1, 0], :, :], (1, 2, 0))
    img_bic = (img_bic * 255.0).round()

    # SR model
    img_bic = img_bic * 1.0 / 255
    img_bic = torch.from_numpy(np.transpose(
        img_bic[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_bic = img_bic.unsqueeze(0)
    img_bic = img_bic.to(device)

    img_sr = r2b_model(img_bic).data.squeeze(
    ).float().cpu().clamp_(0, 1).numpy()
    img_sr = np.transpose(img_sr[[2, 1, 0], :, :], (1, 2, 0))
    img_sr = (img_sr * 255.0).round()

    cv2.imwrite(
        os.path.join(output_dir, '{:s}.png'.format(base)), img_sr)
