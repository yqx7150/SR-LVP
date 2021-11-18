import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from model.model import InvISPNet
from dataset.mri_dataset import mriDataset, mriDataset12, mriDataset12_real_imag, mriDataset12_real_imag_cross
from config.config import get_arguments
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim
from scipy.io import savemat
from matplotlib import pyplot as plt

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')
parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
args = parser.parse_args()

ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
if args.split_to_patch:
    os.makedirs(args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name)
else:
    os.makedirs(args.out_path+"%s/results_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_%s/"%(args.task, ckpt_name)


def main(args):
    net = InvISPNet(channel_in=24, channel_out=24, block_num=8)
    device = torch.device("cuda:0")
    net.to(device)
    net.eval()
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    
    print("[INFO] Start data load and preprocessing") 
    mri_dataset = mriDataset12_real_imag_cross(root1='/home/b110/文档/IISP/data_brain/test_data/test_12ch',root='/home/b110/文档/IISP/data_brain/test_data/test_12ch')
    dataloader = DataLoader(mri_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True) 
    
    print("[INFO] Start test...") 
    # PSNR_COMPRESS = []
    # SSIM_COMPRESS = []
    # PSNR_DECOMPRESS = []
    # SSIM_DECOMPRESS = []
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time() 
        
        input_channel24_mri_real_imag = sample_batched['input_channel24_mri'].to(device)
        # target_channel8x3_mri_real_imag = sample_batched['target_channel8x3_mri'].to(device)
        target_channel24_mri_real_imag = sample_batched['target_channel24_mri'].to(device)
        channelname = int(sample_batched['channelname'][0])

        with torch.no_grad():
            reconstruct_compress_real_imag = net(input_channel24_mri_real_imag)
            reconstruct_decompress_real_imag = net(reconstruct_compress_real_imag, rev=True)

        pred_compress_real_imag = reconstruct_compress_real_imag.detach().permute(0,2,3,1).squeeze()
        pred_decompress_real_imag = reconstruct_decompress_real_imag.detach().permute(0,2,3,1).squeeze()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.detach().permute(0,2,3,1).squeeze()
        
        pred_compress_real_imag = pred_compress_real_imag.cpu().numpy()
        pred_decompress_real_imag = pred_decompress_real_imag.cpu().numpy()
        target_channel24_mri_real_imag = target_channel24_mri_real_imag.cpu().numpy()

        ori_complex = np.zeros([256, 256, 12], dtype=np.complex64)
        for i in range(0, 24, 2):
            ori_complex[:, :, int(i/2)] = target_channel24_mri_real_imag[:, :, i] + 1j*target_channel24_mri_real_imag[:, :, i+1]
        savemat('./mat_results/ori_{}.mat'.format(channelname), {'Img':ori_complex})
        # ori_complex_sos = np.sqrt(np.sum(np.abs(ori_complex)**2, axis=2))

        pred_decompress_complex = np.zeros([256, 256, 12], dtype=np.complex64)
        for i in range(0, 24, 2):
            pred_decompress_complex[:, :, int(i/2)] = pred_decompress_real_imag[:, :, i] + 1j*pred_decompress_real_imag[:, :, i+1]
        savemat('./mat_results/backward_{}.mat'.format(channelname), {'Img':pred_decompress_complex})
        # pred_decompress_complex_sos = np.sqrt(np.sum(np.abs(pred_decompress_complex)**2, axis=2))
        
        pred_compress_complex = np.zeros([256, 256, 4], dtype=np.complex64)
        for i in range(0, 8, 2):
            pred_compress_complex[:, :, int(i/2)] = pred_compress_real_imag[:, :, i] + 1j*pred_compress_real_imag[:, :, i+1]
        savemat('./mat_results/forward_{}.mat'.format(channelname), {'Img':pred_compress_complex})
        # pred_compress_complex_sos = np.sqrt(np.sum(np.abs(pred_compress_complex)**2, axis=2))
    #     print(abs(255*abs(ori_complex_sos) - 255*abs(pred_compress_complex_sos)))
    #     plt.subplot(2 ,2, 1)
    #     plt.imshow(255*abs(ori_complex_sos) ,cmap='gray')
    #     plt.title("ori_12_channel_sos")
    #     plt.subplot(2 ,2, 2)
    #     plt.imshow(255*abs(pred_compress_complex_sos) ,cmap='gray')
    #     plt.title("compress_4_channel_sos")
    #     plt.subplot(2 ,2, 3)
    #     plt.imshow(abs(255*abs(ori_complex_sos) - 255*abs(pred_compress_complex_sos)) ,cmap='gray')
    #     plt.title("error")
    #     plt.subplot(2 ,2, 4)
    #     plt.imshow(255*abs(pred_decompress_complex_sos) ,cmap='gray')
    #     plt.title("decompress_12_channel_sos")
    #     #plt.ion()
    #     #plt.pause(1.5)
    #     #plt.close()
    #     plt.show()
    #     #print(np.max(ori_complex_sos))
    #     #print(np.min(ori_complex_sos))
    #     #print(np.max(pred_compress_complex_sos))
    #     #print(np.min(pred_compress_complex_sos))

    #     psnr_compress = compare_psnr(255*abs(ori_complex_sos), 255*abs(pred_compress_complex_sos), data_range=255)
    #     ssim_compress = compare_ssim(abs(ori_complex_sos), abs(pred_compress_complex_sos), data_range=1)
    #     print('psnr_compress:',psnr_compress,'    ssim_compress:',ssim_compress)
    #     PSNR_COMPRESS.append(psnr_compress)
    #     SSIM_COMPRESS.append(ssim_compress)

    #     psnr_decompress = compare_psnr(255*abs(ori_complex_sos), 255*abs(pred_decompress_complex_sos), data_range=255)
    #     ssim_decompress = compare_ssim(abs(ori_complex_sos), abs(pred_decompress_complex_sos), data_range=1)
    #     print('psnr_decompress:',psnr_decompress,'    ssim_decompress:',ssim_decompress)
    #     PSNR_DECOMPRESS.append(psnr_decompress)
    #     SSIM_DECOMPRESS.append(ssim_decompress)

        del reconstruct_compress_real_imag
        del reconstruct_decompress_real_imag
    # ave_psnr_compress = sum(PSNR_COMPRESS) / len(PSNR_COMPRESS)
    # ave_ssim_compress = sum(SSIM_COMPRESS) / len(SSIM_COMPRESS)
    # ave_psnr_decompress = sum(PSNR_DECOMPRESS) / len(PSNR_DECOMPRESS)
    # ave_ssim_decompress = sum(SSIM_DECOMPRESS) / len(SSIM_DECOMPRESS)
    # print("ave_psnr_compress: %.10f || ave_psnr_decompress:%.10f"%(ave_psnr_compress, ave_psnr_decompress))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

