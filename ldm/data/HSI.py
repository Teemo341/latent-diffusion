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
                 label_root,
                 size=None,
                 split = None,
                 split_rate = 0.8,
                 value_range = None,
                 augment = True,
                 ):
        #data_root: dir of .mat file
        #size: hw, None means whole image
        #split: 'train' means training dataset , 'valid' means validation, None means all for training.
        #split_rate: how many images for training
    
        self.whole_image = self.read_data(data_root)
        self.whole_label = self.read_label(label_root)
        if value_range:
            self.max_value = value_range
            self.whole_image = np.clip(self.whole_image,0,value_range)
        else: # delete outlier
            self.max_value = self.whole_image.max()
        self.size = size
        self.split = split
        self.split_rate = split_rate
        if augment:
            self.augment_method = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(90),
            ])
        else:
            self.augment_method = None
        assert self.size ==None or (self.size <= self.whole_image.shape[0] and self.size <= self.whole_image.shape[1])

    def read_data(self, data_root):
        # read the whole HSI image,reutrun h*w*c, 20*20*200 for example
        return np.array(torch.randn(20, 20, 200))
    
    def read_label(self, label_root):
        # read the whole HSI label one-hot encoding, return h*w*label, 20*20*17 for example
        return np.array(torch.randn(20, 20, 17))

    def __len__(self):
        # calculate the center pixel numbers (w-size+1)*(h-size+1)
        if self.size == None:
            return 1
        else:
            h, w, c = self.whole_image.shape
            length = (w-self.size+1)*(h-self.size+1)
            if not self.split:
                return length
            else:
                if self.split == 'train':
                    return int(length*self.split_rate)
                elif self.split == 'valid':
                    self.drop_length = int(length*self.split_rate)
                    return length - self.drop_length

    def __getitem__(self, i):
        example = dict()
        if self.size == None:
            image = self.whole_image
            label = self.whole_label
        else:
            if self.split == 'valid':
                i = i+ self.drop_length
            h, w, c = self.whole_image.shape
            w_ = w-self.size+1
            image_start_point_h = int(i//w_)
            image_start_point_w = int(i%w_)
            image = self.whole_image[image_start_point_h:(image_start_point_h+self.size),image_start_point_w:(image_start_point_w+self.size),:]
            label = self.whole_label[image_start_point_h:(image_start_point_h+self.size),image_start_point_w:(image_start_point_w+self.size),:]

        # default to score-sde preprocessing
        img = np.array(image)
        lbl = np.array(label)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        lbl = lbl[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        image = torch.tensor(img)
        image = image/self.max_value*255
        label = torch.tensor(lbl)

        image_channel = image.shape[2]
        image_label = torch.cat([image, label], dim=2)

        image_label = rearrange(image_label, 'h w c -> c h w')
        if self.augment_method:
            image_label = self.augment_method(image_label)
        image_label = rearrange(image_label, 'c h w -> h w c')
        image = image_label[:,:,:image_channel]
        label = image_label[:,:,image_channel:]

        image = np.array(image)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["label"] = np.array(label).astype(np.float32)
        return example
    
    def get_max(self):
        return self.max_value


class Indian_Pines_Corrected(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/Indian_pines_corrected.mat", label_root="data/HSI/Indian_pines_gt.mat",value_range=8192, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)["indian_pines_corrected"]
        whole_image = np.array(whole_image,dtype=np.float32)
        #(145, 145, 200).[36,17,11]
        return whole_image
    
    def read_label(self, label_root):
        whole_label = loadmat(label_root)['indian_pines_gt']
        whole_label_ = np.zeros((whole_label.shape[0], whole_label.shape[1], whole_label.max()+1))
        for i in range(whole_label.max()+1):
            whole_label_[:,:,i] = whole_label==i
        # 16 class, 0 means background, one-hot encoding (145, 145, 17)
        return whole_label_
class Indian_Pines_Corrected_train(Indian_Pines_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)
class Indian_Pines_Corrected_valid(Indian_Pines_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='valid', **kwargs)


class KSC_Corrected(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/KSC_corrected.mat", label_root="data/HSI/KSC_gt.mat",value_range=512, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['KSC']
        whole_image = np.array(whole_image,dtype=np.float32)
        #(512, 614, 176).[28,9,10]
        return whole_image   
    
    def read_label(self, label_root):
        whole_label = loadmat(label_root)['KSC_gt']
        whole_label_ = np.zeros((whole_label.shape[0], whole_label.shape[1], whole_label.max()+1))
        for i in range(whole_label.max()+1):
            whole_label_[:,:,i] = whole_label==i
        # 13 class, 0 means background, one-hot encoding (512, 614, 14)
        return whole_label_
class KSC_Corrected_train(KSC_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)
class KSC_Corrected_valid(KSC_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='valid', **kwargs)


class Pavia(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/Pavia.mat", label_root="data/HSI/Pavia_gt.mat",value_range=4096, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['pavia']
        whole_image = np.array(whole_image,dtype=np.float32)
        #((1096, 715, 102).[46,27,10]
        return whole_image   
    
    def read_label(self, label_root):
        whole_label = loadmat(label_root)['pavia_gt']
        whole_label_ = np.zeros((whole_label.shape[0], whole_label.shape[1], whole_label.max()+1))
        for i in range(whole_label.max()+1):
            whole_label_[:,:,i] = whole_label==i
        # 9 class, 0 means background, one-hot encoding (1096, 715, 10)
        return whole_label_
class Pavia_train(Pavia):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)
class Pavia_valid(Pavia):
    def __init__(self, **kwargs):
        super().__init__(split='valid', **kwargs)


class PaviaU(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/PaviaU.mat", label_root="data/HSI/PaviaU_gt.mat",value_range=4096, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['paviaU']
        whole_image = np.array(whole_image,dtype=np.float32)
        #(610, 340, 103).[46,27,10]
        return whole_image   
    
    def read_label(self, label_root):
        whole_label = loadmat(label_root)['paviaU_gt']
        whole_label_ = np.zeros((whole_label.shape[0], whole_label.shape[1], whole_label.max()+1))
        for i in range(whole_label.max()+1):
            whole_label_[:,:,i] = whole_label==i
        # 9 class, 0 means background, one-hot encoding (610, 340, 10)
        return whole_label_

class PaviaU_train(PaviaU):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)
class PaviaU_valid(PaviaU):
    def __init__(self, **kwargs):
        super().__init__(split='valid', **kwargs)


class Salinas_Corrected(HSIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/HSI/Salinas_corrected.mat", label_root="data/HSI/Salinas_gt.mat",value_range=4096, **kwargs)

    def read_data(self, data_root):
        whole_image = loadmat(data_root)['salinas_corrected']
        whole_image = np.array(whole_image,dtype=np.float32)
        #(512, 217, 204).[36,17,11]
        return whole_image
    
    def read_label(self, label_root):
        whole_label = loadmat(label_root)['salinas_gt']
        whole_label_ = np.zeros((whole_label.shape[0], whole_label.shape[1], whole_label.max()+1))
        for i in range(whole_label.max()+1):
            whole_label_[:,:,i] = whole_label==i
        # 16 class, 0 means background, one-hot encoding (512, 217, 17)
        return whole_label_
class Salinas_Corrected_train(Salinas_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)
class Salinas_Corrected_valid(Salinas_Corrected):
    def __init__(self, **kwargs):
        super().__init__(split='valid', **kwargs)


if __name__=='__main__':
    # a = Indian_Pines_Corrected()
    # image = a.whole_image
    # image = image.reshape(-1)
    # plt.figure(1)
    # plt.hist(image, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.savefig('here.png')

    a = Indian_Pines_Corrected(size = 32)
    print(a[0]['image'].shape)
    print(a[0]['label'].shape)
    a = KSC_Corrected(size = 32)
    print(a[0]['image'].shape)
    print(a[0]['label'].shape)
    a = Pavia(size = 32)
    print(a[0]['image'].shape)
    print(a[0]['label'].shape)
    a = PaviaU(size = 32)
    print(a[0]['image'].shape)
    print(a[0]['label'].shape)
    a = Salinas_Corrected(size = 32)
    print(a[0]['image'].shape)
    print(a[0]['label'].shape)