# 该训练代码为训练2通道图像转向为12通道图像，进行医学图像在图像域的超分辨率重建
import numpy as np
import os, time, random
import argparse
import json

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from model.model import InvISPNet
#from dataset.FiveK_dataset import FiveKDatasetTrain
from dataset.mri_dataset import mriDataset12and4_real_imag_cross
from config.config import get_arguments
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr, compare_ssim
from scipy.io import savemat
from matplotlib import pyplot as plt

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))   #求剩余容量最多的显卡
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

parser = get_arguments()   #参数传递
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")   #模型保存路径
# 这里store_true的作用
# 不加--resume，默认为False
# 如果有default，则当不加--resume，默认为default设定的值
parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")    #恢复训练
parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")   #损失函数
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")     #学习率
parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")    #数据增强
args = parser.parse_args()  #解析参数
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)   #exist_ok=True当前目录存在不抛出异常，递归创建目录
os.makedirs(args.out_path+"%s"%args.task, exist_ok=True)
os.makedirs(args.out_path+"%s/checkpoint"%args.task, exist_ok=True)  #这两行是什么，创建子目录吗？？？？？为什么连接符是加号

with open(args.out_path+"%s/commandline_args.yaml"%args.task , 'w') as f:   #文件写操作      commandline_args.yaml这个文件是干嘛的？？？？？
    json.dump(args.__dict__, f, indent=2)   #输出json格式，对数据进行编码   __dict__是什么类，存的是输出的属性把？？？？？？
    
#writer = SummaryWriter('./logs')

def sos_torch_24(img_tensor):
    # # 全部平方
    # [1, 24, 256, 256] = img_tensor**2
    # # 一对相加 变为24通道
    # [1, 24, 256, 256] = plus()
    # # sqrt + 平方
    # [1, 24, 256, 256] = (sqrt())**2
    # # sum / 2
    # [1, 1, 256, 256] = sum(axis=1)/2
    # # sqrt
    # [1, 1, 256, 256] = sqrt()
    tensor_square = torch.FloatTensor(1, 24, 256, 256)   #创建个（1，24，256，256）的浮点tensor
    tensor_square = torch.mul(img_tensor, img_tensor)   #对两个张量进行逐元素乘法
    tensor_sum = torch.FloatTensor(1, 12, 256, 256)  #创建个（1，12，256，256）的浮点tensor
    for i in range(0, int(2*tensor_sum.size()[1]), 2):
        tensor_sum[:, int(i/2), :, :] = tensor_square[:, i, :, :] + tensor_square[:, i+1, :, :]    #实部和虚部合成12通道复数图
    tensor_sum_all = torch.FloatTensor(1, 1, 256, 256)   #创建个（1，1，256，256）的浮点tensor
    tensor_sum_all = torch.sum(tensor_sum, dim=1)     #dim=1横向求和，12通道求和
    tensor_sqrt = torch.FloatTensor(1, 1, 256, 256)
    tensor_sqrt = torch.sqrt(tensor_sum_all)
    tensor_sqrt = tensor_sqrt.cuda()
    
    return tensor_sqrt

def sos_torch_8(img_tensor, st):
    # # 全部平方
    # [1, 24, 256, 256] = img_tensor**2
    # # 一对相加 变为24通道
    # [1, 24, 256, 256] = plus()
    # # sqrt + 平方
    # [1, 24, 256, 256] = (sqrt())**2
    # # sum / 2
    # [1, 1, 256, 256] = sum(axis=1)/2
    # # sqrt
    # [1, 1, 256, 256] = sqrt()
    tensor_square = torch.FloatTensor(1, 24, 256, 256).cuda()
    tensor_square = torch.mul(img_tensor, img_tensor)
    tensor_sum = torch.FloatTensor(1, 2, 256, 256).cuda()
    for i in range(0, int(2*tensor_sum.size()[1]), 2):
        tensor_sum[:, int(i/2), :, :] = torch.add(tensor_square[:, i+int(st), :, :], tensor_square[:, i+1+int(st), :, :])
    tensor_sum_all = torch.FloatTensor(1, 1, 256, 256).cuda()
    tensor_sum_all = torch.sum(tensor_sum, dim=1)
    tensor_sqrt = torch.FloatTensor(1, 1, 256, 256).cuda()
    tensor_sqrt = torch.sqrt(tensor_sum_all)
    tensor_sqrt = tensor_sqrt.cuda()
    
    return tensor_sqrt

def main(args):
    # 设置网络
    net = InvISPNet(channel_in=24, channel_out=24, block_num=8)
    # 将网络加载到cuda上
    net.cuda()
    # 如果有之前训练0了的模型，可以先加载已保存的网络权重，之后再训练
    if args.resume:
        net.load_state_dict(torch.load(args.out_path+"%s/checkpoint/latest.pth"%args.task))
        print("[INFO] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)
    # 设置优化器，这里可以设置成其它优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.5)     #调整学习率
    
    print("[INFO] Start data loading and preprocessing")
    # 数据集加载
    mri_Dataset = mriDataset12and4_real_imag_cross(root1='/home/b110/文档/IISP/data_brain/train_12coil/train_12ch', root2='/home/b110/文档/IISP/data_brain/train_2coil_SCC', root='/home/b110/文档/IISP/data_brain/train_12coil/train_12ch')
    dataloader = DataLoader(mri_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    #mri_Dataset_val = mriDataset12_real_imag_cross(root1='./data/train_12coil/val_12ch',root2='./data/train_4coil/val_4ch',root='./data/train_12coil/val_12ch')        
    #dataloader_val = DataLoader(mri_Dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    print("[INFO] Start to train")
    step = 0
    for epoch in range(0, 5000):  #每张图训练5000次
        epoch_time = time.time()
        # train
        i = 0
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()    #时间
            # 输入，正向输出label，反向输出label  
            input_c12_mri, output_c12_mri, input_c8x3_mri = sample_batched['input_channel24_mri'].cuda(), sample_batched['target_channel24_mri'].cuda(), sample_batched['input_channel8x3_mri'].cuda()
            # 将输入进入网络中
            reconstruct_c12 = net(input_c8x3_mri)
            # 这是正向的loss为l1_loss
            # compress_loss = F.l1_loss(reconstruct_c4, c4_mri)
            # 这是正向的sos_loss，为l1_loss
            sos_reconstruct_c12 = sos_torch_24(reconstruct_c12)
            # log tensorboard
            # writer.add_image('train_compress_epoch_{}'.format(epoch), sos_reconstruct_c4, global_step=i, dataformats='CHW')
            sos_output_c12_mri = sos_torch_24(input_c12_mri)
            # log tensorboard
            # writer.add_image('train_ori_epoch_{}'.format(epoch), sos_input_c12_mri, global_step=i, dataformats='CHW')
            sos_forward_loss = F.smooth_l1_loss(sos_reconstruct_c12, sos_output_c12_mri)
            #sos_forward_loss = F.l1_loss(sos_reconstruct_c12, sos_output_c12_mri)
            #print("sos_compress_loss:  %.10f"%(sos_compress_loss.detach().cpu().numpy()))
            # 反向传播
            reconstruct_c4_rev = net(reconstruct_c12, rev=True)
            # 这是反向的loss为l1_loss
            #decompress_loss = F.smooth_l1_loss(reconstruct_c12, output_c12_mri)
            sos_reconstruct_c4_rev = sos_torch_8(reconstruct_c4_rev, 0)
            sos_reconstruct_c4_rev_2 = sos_torch_8(reconstruct_c4_rev, 4)
            sos_reconstruct_c4_rev_3 = sos_torch_8(reconstruct_c4_rev, 8)
            sos_reconstruct_c4_rev_4 = sos_torch_8(reconstruct_c4_rev, 12)
            sos_reconstruct_c4_rev_5 = sos_torch_8(reconstruct_c4_rev, 16)
            sos_reconstruct_c4_rev_6 = sos_torch_8(reconstruct_c4_rev, 20)
            sos_output_c12_mri_rev = sos_torch_24(output_c12_mri)
            #rev_loss = (F.l1_loss(sos_reconstruct_c4_rev, sos_output_c12_mri_rev) + F.l1_loss(sos_reconstruct_c4_rev_2, sos_output_c12_mri_rev) + F.l1_loss(sos_reconstruct_c4_rev_3, sos_output_c12_mri_rev)) / 3
            rev_loss = (F.smooth_l1_loss(sos_reconstruct_c4_rev, sos_output_c12_mri_rev) + F.smooth_l1_loss(sos_reconstruct_c4_rev_2, sos_output_c12_mri_rev) + F.smooth_l1_loss(sos_reconstruct_c4_rev_3, sos_output_c12_mri_rev) + F.smooth_l1_loss(sos_reconstruct_c4_rev_4, sos_output_c12_mri_rev) + F.smooth_l1_loss(sos_reconstruct_c4_rev_5, sos_output_c12_mri_rev) + F.smooth_l1_loss(sos_reconstruct_c4_rev_6, sos_output_c12_mri_rev)) / 6
            # 正向loss和反向loss加权求和
            loss = args.forward_weight * sos_forward_loss + rev_loss   #反向也是12通道的作为标签的啊？？？？？
            # 反传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            print("Epoch: %d Step: %d || loss: %.10f sos_forward_loss:  %.10f rev_loss: %.10f || lr: %f time: %f"%(
                epoch, step, loss.detach().cpu().numpy(),  
                sos_forward_loss.detach().cpu().numpy(), rev_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
            )) 
            step += 1
            i = i + 1
        # 保存模型
        torch.save(net.state_dict(), args.out_path+"%s/checkpoint/latest.pth"%args.task)
        if (epoch+1) % 1 == 0:
            # 这是每1个epoch保存一次模型
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(net.state_dict(), args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))  #这是保存网络参数把，不是保存整个模型把？？？？？
            print("[INFO] Successfully saved "+args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
        # 学习率下降
        scheduler.step()   
        
        print("[INFO] Epoch time: ", time.time()-epoch_time, "task: ", args.task)
        

if __name__ == '__main__':
    # 设置 PyTorch 进行 CPU 多线程并行计算时所占用的线程数，用来限制 PyTorch 所占用的 CPU 数目
    torch.set_num_threads(4)
    main(args)
