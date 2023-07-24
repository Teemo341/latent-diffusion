import os
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat


class HSIBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 value_range = None,
                 flip_p=0.5
                 ):
        self.whole_image = self.read_data(data_root)
        self.size = size
        self.value_range = value_range
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        assert self.size ==None or (self.size <= self.whole_image.shape[0] and self.size <= self.whole_image.shape[1])

    def read_data(self, data_root):
        # read the whole HSI image,reutrun h*w*c, 20*20*200 for example
        return torch.randn(20, 20, 200)

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
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = torch.tensor(img)
        if self.value_range ==None:
            min_value = image.min()
            max_value = image.max()
            example["value_min"]=min_value
            example["value_max"]=max_value
            image = (image-min_value)/max_value*255
        else:
            min_value = 0
            max_value = self.value_range
            example["value_min"]=min_value
            example["value_max"]=max_value
            image = (image-min_value)/max_value*255

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class Indian_Pines_Corrected(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/Indian_pines_corrected.mat", **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['indian_pines_corrected']
        whole_image = np.array(whole_image,dtype=np.float32)
        #(145, 145, 200)
        return whole_image
