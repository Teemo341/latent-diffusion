import os
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
from einops import rearrange
import matplotlib.pyplot as plt


class HSIBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 value_range = None,
                 flip_p=0.5
                 ):
        self.whole_image = self.read_data(data_root)
        if value_range:
            self.max_value = value_range
            self.whole_image = np.clip(self.whole_image,0,value_range)
        else: # delete outlier
            self.max_value = self.whole_image.max()
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        assert self.size ==None or (self.size <= self.whole_image.shape[0] and self.size <= self.whole_image.shape[1])

    def read_data(self, data_root):
        # read the whole HSI image,reutrun h*w*c, 20*20*200 for example
        return np.array(torch.randn(20, 20, 200))

    def __len__(self):
        # calculate the center pixel numbers (w-size+1)*(h-size+1)
        if self.size == None:
            return 1
        else:
            h, w, c = self.whole_image.shape
            return (w-self.size+1)*(h-self.size+1)

    def __getitem__(self, i):
        example = dict()
        if self.size == None:
            image = self.whole_image
        else:
            h, w, c = self.whole_image.shape
            w_ = w-self.size+1
            image_start_point_h = int(i//w_)
            image_start_point_w = int(i%w_)
            image = self.whole_image[image_start_point_h:(image_start_point_h+self.size),image_start_point_w:(image_start_point_w+self.size),:]

        # default to score-sde preprocessing
        img = np.array(image)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = torch.tensor(img)
        image = image/self.max_value*255

        image = rearrange(image, 'h w c -> c h w')
        image = self.flip(image)
        image = rearrange(image, 'c h w -> h w c')
        image = np.array(image)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    
    def get_max(self):
        return self.max_value


class Indian_Pines_Corrected(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/Indian_pines_corrected.mat",value_range=8192, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['indian_pines_corrected']
        whole_image = np.array(whole_image,dtype=np.float32)
        #(145, 145, 200)
        return whole_image


if __name__=='__main__':
    a = Indian_Pines_Corrected()
    image = a.whole_image
    image = image.reshape(-1)
    plt.figure(1)
    plt.hist(image, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.savefig('here.png')