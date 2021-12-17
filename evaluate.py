
import os
import math
import numpy as np
import cv2
import glob

from R2B_Network.utils.metrics import PSNR, SSIM, bgr2ycbcr


HR_dir = '/content/drive/MyDrive/outputs/r2b/realsr_2lakh'
SR_dir = '/content/drive/MyDrive/Colab Notebooks/l2lRaGan/dataset/Test/RealSR/LR'

crop_border = 4
test_Y = True  # True: test Y channel only; False: test RGB channels

PSNR_all = []
SSIM_all = []
img_list = sorted(glob.glob(HR_dir + '/*'))

if test_Y:
    print('Testing Y channel.')
else:
    print('Testing RGB channels.')

for i, img_path in enumerate(img_list):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img_HR = cv2.imread(img_path) / 255.
    img_SR = cv2.imread(os.path.join(
        SR_dir, base_name+'.png')) / 255.

    # evaluate on Y channel in YCbCr color space
    if test_Y and img_HR.shape[2] == 3:
        img_HR = bgr2ycbcr(img_HR)
        img_SR = bgr2ycbcr(img_SR)
    else:
        img_HR = img_HR
        img_SR = img_SR

    # crop borders
    if img_HR.ndim == 3:
        cropped_GT = img_HR[crop_border:-
                            crop_border, crop_border:-crop_border, :]
        cropped_Gen = img_SR[crop_border:-
                             crop_border, crop_border:-crop_border, :]
    elif img_HR.ndim == 2:
        cropped_HR = img_HR[crop_border:-
                            crop_border, crop_border:-crop_border]
        cropped_SR = img_SR[crop_border:-
                            crop_border, crop_border:-crop_border]
    else:
        raise ValueError(
            'Wrong image dimension: {}. Should be 2 or 3.'.format(img_HR_in.ndim))

    # calculate PSNR and SSIM
    psnr = PSNR(cropped_HR * 255, cropped_SR * 255)

    ssim = SSIM(cropped_HR * 255, cropped_SR * 255)
    print('{:3d} - {:15}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
        i + 1, base_name, psnr, ssim))
    PSNR_all.append(psnr)
    SSIM_all.append(ssim)
print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
    sum(PSNR_all) / len(PSNR_all),
    sum(SSIM_all) / len(SSIM_all)))
