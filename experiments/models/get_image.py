# 导入torch模块
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange
from ldm.data.HSI import *
import numpy as np


# 加载数据集
def get_whole_image(name): # b h w c
    if name == 'Indian_Pines_Corrected':
        dataset = Indian_Pines_Corrected_train()
        #(145, 145, 200).[36,17,11]
    elif name == 'KSC_Corrected':
        dataset = KSC_Corrected()
        #(512, 614, 176).[28,9,10]
    elif name == "Pavia":
        dataset = Pavia()
        #((1093, 715, 102).[46,27,10]
    elif name == "PaviaU":
        dataset = PaviaU()
        #(610, 340, 103).[46,27,10]
    elif name == "Salinas_Corrected":
        dataset = Salinas_Corrected()
        #(512, 217, 204).[36,17,11]
    else:
        raise ValueError("Unsupported dataset")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=4)
    return dataloader


# 得到数据集对应图像的维度
def get_dim(name):
    dic = {}
    dic["Indian_Pines_Corrected"] = 200
    dic["KSC_Corrected"] = 176
    dic["Pavia"] = 102
    dic["PaviaU"] = 103
    dic["Salinas_Corrected"] = 204
    return dic[name]


# 可视化HSI
class HSI_visualization():
    def __init__(self, name):
        super(HSI_visualization, self).__init__()
        if name == 'Indian_Pines_Corrected':
            self.vis_channels = [36,17,11]
        elif name == 'KSC_Corrected':
            self.vis_channels = [28,9,10]
        elif name == "Pavia":
            self.vis_channels =[46,27,10]
        elif name == "PaviaU":
            self.vis_channels =[46,27,10]
        elif name == "Salinas_Corrected":
            self.vis_channels =[36,17,11]
        else:
            raise ValueError("Unsupported dataset")

    def visualization(self, HSI_image):
        RGB_image = HSI_image[:,:,self.vis_channels]
        RGB_image = (RGB_image+1.0)/2.0 # 将数据归一化到[0,1]
        RGB_image = (RGB_image.numpy() * 255).astype(np.uint8)
        return RGB_image
    
# 生成HSI
def sample(generator, name, sample_times=8, save_full = True, save_RGB = True, h=32, w=32, save_dic="./results/GAN"):
    save_dic = os.path.join(save_dic, name)
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)

    with torch.no_grad():
        z = torch.randn(sample_times, get_dim(name), h, w).to(device)
        generated_images = generator(z).cpu()
        generated_images = rearrange(generated_images, 'b c h w -> b h w c')

    #保存为npy
    if save_full:
        for i in range(sample_times):
            np.save(os.path.join(save_dic, f"generated_{i}.npy"), generated_images[i].numpy())

    # 保存伪彩色
    if save_RGB:
        vis = HSI_visualization(name)
        for i in range(sample_times):
            image = vis.visualization(generated_images[i])
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(os.path.join(save_dic, f"generated_{i}.png"),dpi=100,bbox_inches='tight',pad_inches = 0)
            # plt.imsave(os.path.join(save_dic, f"generated_{i}.png"), image,dpi=100,bbox_inches='tight',pad_inches = 0)
    return generated_images

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--datasets', type=str, nargs='+', default=['Indian_Pines_Corrected', 'KSC_Corrected', 'Pavia', 'PaviaU', 'Salinas_Corrected'], help='which datasets, default all')
    paser.add_argument('--image_dir', type=str, default='./experiments/metric/original_image', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        dataloader = get_whole_image(name)
        for i, data in enumerate(dataloader):
            if i == 0:
                data = data["image"]
                save_dic = os.path.join(args.image_dir, name)
                if not os.path.exists(save_dic):
                    os.makedirs(save_dic)
                np.save(os.path.join(save_dic, f"hsi.npy"), data[0].numpy())
                vis = HSI_visualization(name)
                image = vis.visualization(data[0])
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(os.path.join(save_dic, f"RGB.png"),dpi=100,bbox_inches='tight',pad_inches = 0)
                break
    