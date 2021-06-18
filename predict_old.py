import numpy as np
import os
import shutil
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt

from config import update_config
from Dataset import Label_loader
from utils import psnr_error
import Dataset
from models.unet import UNet

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default='latest_avenue_40001.pth', type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', default=True, action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--show_heatmap', default=True, action='store_true',
                    help='Show and save the difference heatmap real-timely, this drops fps.')


def val(cfg, model=None):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    fps = 0
    psnr_group = []

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset_use = Dataset.predict_dataset(cfg, folder)
            psnrs = []
            folder_name = folder.split('/')[-1]
            save_path = './results/' + folder_name
            if os.path.exists(save_path):
                pass
            else:
                os.mkdir(save_path)
            for j, clip in enumerate(dataset_use):
                input_np = clip['video_clip'][0:12, :, :]
                target_np = clip['video_clip'][12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()

                G_frame = generator(input_frames)
                test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

                # print(input_np.shape)
                if float(test_psnr) < 30:
                    pic_name = os.path.split(clip['video_name'][-1])[1]

                    shutil.copyfile(clip['video_name'][-1], os.path.join(save_path,pic_name))
                    # cv2.imwrite('results/'+str(i)+str(j)+'.png', target_np.transpose(1,2,0)) #input_np[0:3].transpose(1,2,0)[...,::-1]

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                
                print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset_use)}, {fps:.2f} fps.', end='')
            # print(psnrs)


    print('\nAll frames were detected, begin to compute AUC.')

if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='predict')
    test_cfg.print_cfg()
    val(test_cfg)