# 导入torch模块
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange
from ldm.data.HSI import *
import numpy as np
from torchsummary import summary
from torch.autograd import Variable

img_shape = (200, 32, 32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.reshape(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.reshape(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity




# ----------
#  Training
# ----------

batches_done = 0
def train(generator, discriminator, train_loader, num_epochs=10):
    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95) 
    for epoch in range(num_epochs):

        for i, input_dic in enumerate(train_loader):
            real_images = input_dic["image"]
            # real_images = torch.tensor(real_images)
            real_images = rearrange(real_images, 'b h w c -> b c h w')
            real_images = real_images.to(device)

            generator.eval()
            discriminator.train()
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            z = torch.randn(real_images.shape[0], 100, dtype=torch.float32).to(device) # 生成随机噪声
            fake_imgs = generator(z).detach().to(device)
            loss_D = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_imgs))
            loss_D.backward()
            optimizer_D.step()
            # 计算损失
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)


            if i % 5 == 0:
                generator.train()
                discriminator.eval()
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                # Generate a batch of images
                gen_imgs = generator(z).to(device)
                loss_G = -torch.mean(discriminator(gen_imgs))
                loss_G.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d] [D loss: %f] [G loss: %f]"
                    % (epoch, loss_D.item(), loss_G.item())
                )

            if (epoch+1) % 1 == 0:
                sample(generator, name, sample_times=8, save_full=False, save_RGB=True, h=32, w=32, save_dic=f"./experiments/models/WGAN/Indian_Pines_Corrected/{epoch+1}")
        scheduler_G.step()   
        scheduler_D.step() 


def get_dataloader(name,batch_size,image_size): # b h w c
    if name == 'Indian_Pines_Corrected':
        dataset = Indian_Pines_Corrected(size = image_size)
        #(145, 145, 200).[36,17,11]
    elif name == 'KSC_Corrected':
        dataset = KSC_Corrected(size = image_size)
        #(512, 614, 176).[28,9,10]
    elif name == "Pavia":
        dataset = Pavia(size = image_size)
        #((1093, 715, 102).[46,27,10]
    elif name == "PaviaU":
        dataset = PaviaU(size = image_size)
        #(610, 340, 103).[46,27,10]
    elif name == "Salinas_Corrected":
        dataset = Salinas_Corrected(size = image_size)
        #(512, 217, 204).[36,17,11]
    else:
        raise ValueError("Unsupported dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)
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
def sample(generator, name, sample_times=8, save_full = True, save_RGB = True, h=32, w=32, save_dic="./results/WGAN"):
    save_dic = os.path.join(save_dic, name)
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        z = torch.randn(sample_times, 100).to(device)
        generated_images = generator(z).cpu()
        generated_images = rearrange(generated_images, 'b c h w -> b h w c')
        generated_images = torch.clamp(generated_images, min=-1.0, max=1.0)

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
    # dataset
    paser.add_argument('--datasets', type=str, nargs='+', default=['Indian_Pines_Corrected', 'KSC_Corrected', 'Pavia', 'PaviaU', 'Salinas_Corrected'], help='which datasets, default all')
    paser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    paser.add_argument('--image_size', type=int, default=32, help='size of the image')
    # model hyperparam
    paser.add_argument('--hidden_dim', type=int, default=32, help='dimensionality of the hidden space')
    paser.add_argument('--layers', type=int, default=9, help='number of res blocks')
    # training hyperparam
    paser.add_argument('--epochs', type=int, default=10, help='number of epochs of training')
    paser.add_argument('--warmup_epoches', type=int, default=0, help='number of warmup epochs of training')
    paser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    paser.add_argument('--sample_times', type=int, default=8, help='number of sample times')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/WGAN/Indian_Pines_Corrected', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/GAN/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            generator = Generator()
            discriminator = Discriminator()
            generator.load_state_dict(torch.load(f"{checkpoint_dir}/generator.pth"))
            discriminator.load_state_dict(torch.load(f"{checkpoint_dir}/discriminator.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            h = w = args.image_size
            generator = Generator()
            discriminator = Discriminator()
            dataloader = get_dataloader(name, args.batch_size, args.image_size)
            train(generator, discriminator, dataloader, num_epochs=args.epochs)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(generator.state_dict(), f"{checkpoint_dir}/generator.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

        sample(generator, name, sample_times=args.sample_times, save_full=args.save_full, save_RGB=args.save_RGB, h=h, w=w, save_dic=args.image_dir)

    print("finish all datasets")

