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

# 定义编解码器
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
        )
 
    def forward(self, x):
        out = self.conv_relu(x)
        out += x
        out = F.leaky_relu(out,inplace=True)
        return out

# 定义生成器
class Generator(nn.Module):
    def __init__(self, img_channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 输入是一个z_dim维的噪声向量
            nn.ConvTranspose2d(100, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        return self.gen(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)


# 定义训练函数
def train(generator, discriminator, train_loader, valid_loader = None, num_epochs=100, warmup_epoches = 0, erly_stop = 5):

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))  # 生成器的优化器
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))  # 判别器的优化器
    #scheduler = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1) #学习率的调节器

    for epoch in range(num_epochs):

        for i, input_dic in enumerate(train_loader):
            data = input_dic["image"].to(device)
            real = data.to(device)
            real = rearrange(real, 'b h w c -> b c h w')
            batch_size = real.size(0)

            # 训练判别器
            generator.eval()
            discriminator.train()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            real_preds = discriminator(real)
            real_loss = criterion(real_preds, real_labels)

            z = torch.randn(batch_size, 100, device=device)
            fake = generator(z)
            fake_preds = discriminator(fake.detach())
            fake_loss = criterion(fake_preds, fake_labels)

            disc_loss = (real_loss + fake_loss) / 2
            optimizer_D.zero_grad()

            disc_loss.backward()
            optimizer_D.step()

            # 训练生成器
            generator.train()
            discriminator.eval()
            output = discriminator(fake)
            gen_loss = criterion(output, real_labels)

            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()

            if (i+1) % int(len(train_loader)/2) == 0: #一轮打印两次
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {disc_loss.item():.4f}, g_loss: {gen_loss.item():.4f}")
    

        if (epoch+1) % 1 == 0:
            sample(generator, name, sample_times=8, save_full=False, save_RGB=True, h=32, w=32, save_dic=f"./experiments/models/GAN1/{epoch+1}")


# 加载数据集
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
    paser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    paser.add_argument('--image_size', type=int, default=32, help='size of the image')
    paser.add_argument('--epochs', type=int, default=20, help='number of epochs of training')
    paser.add_argument('--warmup_epoches', type=int, default=0, help='number of warmup epochs of training')
    paser.add_argument('--sample_times', type=int, default=8, help='number of sample times')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints1', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/GAN1', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/GAN/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            generator = torch.load(torch.load(f"{checkpoint_dir}/generator.pth"))
            discriminator = torch.load(torch.load(f"{checkpoint_dir}/discriminator.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            h = w = args.image_size
            generator = Generator(get_dim(name))
            discriminator = Discriminator(get_dim(name))
            dataloader = get_dataloader(name, args.batch_size, args.image_size)
            train(generator, discriminator, dataloader, num_epochs=args.epochs, warmup_epoches=args.warmup_epoches)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(generator.state_dict(), f"{checkpoint_dir}/generator.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

        sample(generator, name, sample_times=args.sample_times, save_full=args.save_full, save_RGB=args.save_RGB, h=h, w=w, save_dic=args.image_dir)

    print("finish all datasets")
