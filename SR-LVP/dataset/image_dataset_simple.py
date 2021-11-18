from __future__ import print_function, division
import os, random, time
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import rawpy
from glob import glob
from PIL import Image as PILImage
import numbers
from scipy.misc import imread

class imageDataset(Dataset):
    def __init__(self, root):
        self.datanames = np.array([root+"/"+x  for x in os.listdir(root)])
    
    def random_flip(self, image_rgb, image_gray):
        '''
        翻转
        '''
        idx = np.random.randint(2)
        image_rgb = np.flip(image_rgb,axis=idx).copy()
        image_gray = np.flip(image_gray,axis=idx).copy()
        
        return image_rgb, image_gray

    def random_rotate(self, image_rgb, image_gray):
        '''
        旋转
        '''
        idx = np.random.randint(4)
        image_rgb = np.rot90(image_rgb,k=idx)
        image_gray = np.rot90(image_gray,k=idx)

        return image_rgb, image_gray

    def random_crop(self, patch_size, image_rgb, image_gray):
        '''
        裁剪
        '''
        H, W, _ = image_rgb.shape
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))

        patch_image_rgb = image_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        patch_image_gray = image_gray[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

        return patch_image_rgb, patch_image_gray
        
    def aug(self, patch_size, image_rgb, image_gray):
        '''
        数据增强
        '''
        image_rgb, image_gray = self.random_crop(patch_size, image_rgb, image_gray)
        image_rgb, image_gray = self.random_rotate(image_rgb, image_gray)
        image_rgb, image_gray = self.random_flip(image_rgb, image_gray)
        
        return image_rgb, image_gray

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989*r + 0.5870*g + 0.1140*b
        return gray
        
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, index):
        # need 3, 256, 256
        self.patch_size=256
        # 加载彩色图片
        image_rgb = imread(self.datanames[index]) # 4284 2844 3
        # 彩色图像灰色化,并且复制为3通道
        image_gray1 = self.rgb2gray(image_rgb)
        image_gray = np.zeros_like(image_rgb)
        image_gray[:, :, 0] = image_gray1
        image_gray[:, :, 1] = image_gray1
        image_gray[:, :, 2] = image_gray1
        # 数据增强
        image_rgb, image_gray = self.aug(self.patch_size, image_rgb, image_gray)
        # 将图像值归一化
        image_rgb = self.norm_img(image_rgb, 255)
        image_gray = self.norm_img(image_gray, 255)
        input_image_rgb = image_rgb.copy()
        # 将以上图像均转换为tensor形式
        input_image_rgb = self.np2tensor(input_image_rgb)
        image_rgb = self.np2tensor(image_rgb)
        image_gray = self.np2tensor(image_gray)

        sample = {'input_image_rgb':input_image_rgb, 'image_gray':image_gray, 'image_rgb':image_rgb}
        return sample


class imageTestDataset(Dataset):
    def __init__(self, root):
        self.datanames = np.array([root+"/"+x  for x in os.listdir(root)])

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989*r + 0.5870*g + 0.1140*b
        return gray
        
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, index):
        # 加载彩色图片
        image_rgb = imread(self.datanames[index]) # 4284 2844 3
        # 彩色图像灰色化,并且复制为3通道
        image_gray1 = self.rgb2gray(image_rgb)
        image_gray = np.zeros_like(image_rgb)
        image_gray[:, :, 0] = image_gray1
        image_gray[:, :, 1] = image_gray1
        image_gray[:, :, 2] = image_gray1
        # 将图像值归一化
        image_rgb = self.norm_img(image_rgb, 255)
        image_gray = self.norm_img(image_gray, 255)
        input_image_rgb = image_rgb.copy()
        # 将以上图像均转换为tensor形式
        input_image_rgb = self.np2tensor(input_image_rgb)
        image_rgb = self.np2tensor(image_rgb)
        image_gray = self.np2tensor(image_gray)

        sample = {'input_image_rgb':input_image_rgb, 'image_gray':image_gray, 'image_rgb':image_rgb, 'file_name':(self.datanames[index]).split('/')[-1].split('.')[0]}
        return sample
