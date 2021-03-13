import sys
import os.path
import glob
import cv2
import numpy as np
import torch
from networks.generator import Generator
import utils.utils as util

cfg = util.load_yaml('../Configs/Train/config_sr.yml')
model_path = sys.argv[1]
device = torch.device('cuda')
# device = torch.device('cpu')

input_dir = '/content/drive/MyDrive/MajorProject/results_r2b/*'
output_dir = '/content/drive/MyDrive/MajorProject/results_sr/'
util.mkdir(output_dir)

model = Generator(cfg['network_G'])
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(input_dir):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(
        img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(
        os.path.join(output_dir, '{:s}.png'.format(base)), output)
