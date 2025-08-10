import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse

import os
import cv2
import scipy.io as sio

from core.loaddata import HSSampledata
from core.common import *


class decoderAE(nn.Module):
    def __init__(self,input_channels=5, output_channels =128):
        super(decoderAE,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.decoderlayer = nn.Conv2d(in_channels=self.input_channels,out_channels=self.output_channels,kernel_size=(1,1),bias=False)
	
    def forward(self, x):
        de_result = self.decoderlayer(x)
        return de_result

def norm(x):
    x = (x - x.min())/(x.max()-x.min())
    return x * 255.0

def tensor2rgb(tensor, r= 14, g=33, b=53):
    imageR = tensor[:,:,r]
    imageG = tensor[:,:,g]
    imageB = tensor[:,:,b]
    image = cv2.merge([imageR,imageG,imageB])
    image = norm(image)
    return image

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def get_dataset(args):
    if args.dataset_name=='Indian_Pines_Corrected':
        r, g, b = 36,17,11
        channel=200
    elif args.dataset_name=='KSC_Corrected':
        r, g, b = 28,9,10
        channel=176
    elif args.dataset_name=='Pavia':
        r, g, b = 46,27,10
        channel=102
    elif args.dataset_name=='PaviaU':
        r, g, b = 46,27,10
        channel=103
    elif args.dataset_name=='Salinas_Corrected':
        r, g, b = 36,17,11
        channel=204
    else:
        raise ValueError("dataset_name not supported")
    
    return r, g, b, channel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--abundance_path', type=str, default='', help='path of synthesized abundance')
    parser.add_argument('--AE_path', type=str, default='', help='name of checkpoints of the unmixingAE')
    parser.add_argument('--dataset_name', type=str, default='KSC_Corrected', help='dataset name')
    parser.add_argument('--result_path', type=str, default='./experiments/Synthesis/HSI/', help='save dir of final results')
    parser.add_argument('--image_path', type=str, default='./experiments/Synthesis/RGB/', help='save dir of RGB images')
    args = parser.parse_args()

    train_path    = args.abundance_path 
    model_name    = args.AE_path
    result_path   = args.result_path
    image_path    = args.image_path
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    r, g, b, channel = get_dataset(args)

    sample_set = HSSampledata(image_dir=train_path, augment=False)
    sample_loader = DataLoader(sample_set, batch_size=1, num_workers=4, shuffle=False)
    ckpt = torch.load(model_name)["model"]
    decoderweight = ckpt['decoderlayer.weight']
    decoderweight = (decoderweight - decoderweight.min()) / (decoderweight.max() - decoderweight.min())
    net = decoderAE(output_channels=channel)
    model_dict = net.state_dict()
    model_dict['decoderlayer.weight'] = decoderweight
    net.load_state_dict(model_dict)
    net.eval().cuda()
    device = torch.device('cuda')
    print('===> Loading testset')
    print('===> Start testing')
    with torch.no_grad():
        output = []
        test_number = 0
        # loading model
        for i, (abu) in enumerate(sample_loader):
            abu =  abu.to(device)
            abu = (abu+1)/2
            y = net(abu)
            y = y.squeeze().cpu().numpy().transpose(1, 2, 0) # [H, W, C]
            filename = f"generated_{i}"
            abu = abu.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            save_dir = result_path + filename + '.mat'
            numpy_dir = result_path + filename + '.npy'
            rgb_dir = image_path + filename + '.png'

            # sio.savemat(save_dir,{'HSI':y, 'Abu':abu, 'End':decoderweight})
            np.save(numpy_dir, y)
            cv2.imwrite(rgb_dir,tensor2rgb(y, r, g, b))
