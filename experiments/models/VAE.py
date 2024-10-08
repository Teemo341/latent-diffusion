# 导入torch模块
import argparse
import torch
import torch.nn as nn

import torch.optim as optim
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange
from ldm.data.HSI import *
import numpy as np
import torch.nn.functional as F
from torchsummary import summary

#class MidLayer(nn.Module):
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
    
class ConvVAE(nn.Module):
    def __init__(self, in_channels, H, W):
        super(ConvVAE,self).__init__()
        self.H = H  # 存储高度
        self.W = W  # 存储宽度
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            ResidualBlock(32)
        )
        self.fc1 = nn.Linear(32 * (self.H // 8) * (self.W // 8), 256)
        self.fc21 = nn.Linear(256, 20)  # 均值向量
        self.fc22 = nn.Linear(256, 20)  # 方差向量

        # 解码器部分
        self.decoder_fc = nn.Linear(20, 32 * (self.H // 8) * (self.W // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(-1, 32 * (self.H // 8) * (self.W // 8))
        h = F.leaky_relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.decoder_fc(z))
        h = h.reshape(-1, 32, self.H // 8, self.W // 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum') 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp()))
    return BCE, KLD


# 定义训练函数
def train(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    for epoch in range(num_epochs):
        model.train()
        total_lossmse = 0
        total_losskl = 0
        for i, input_dic in enumerate(train_loader):
            data = input_dic["image"].to(device)
            # real_images = torch.tensor(real_images)
            data = rearrange(data, 'b h w c -> b c h w')
            optimizer.zero_grad()
            # 进行前向传播
            recon_batch, mu, logvar = model(data)
            # 计算损失
            # print(data.min())
            loss_BCE, loss_KLD = loss_function(recon_batch, data, mu, logvar)
            loss = loss_KLD + loss_BCE
            # 反向传播
            loss.backward()
            optimizer.step()
        

            total_lossmse += loss_BCE.item()     
            total_losskl += loss_KLD.item()
        scheduler.step()
        # 日志输出
        print(f'Epoch {epoch+1}, MSE Loss: {total_lossmse / len(train_loader.dataset):.4f}, KL Loss: {total_losskl / len(train_loader.dataset):.4f}')

        if (epoch+1) % 5 == 0:
            sample(model, name, sample_times=8, save_full=True, save_RGB=True, save_dic=f"./experiments/models/VAE/Salinas_Corrected/{epoch+1}")
# 生成图像

    # 保存或显示 sample，例如使用 save_image


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
def sample(model, name, sample_times=8, save_full = True, save_RGB = True, save_dic="./results/VAE"):
    save_dic = os.path.join(save_dic, name)
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(sample_times, 20).to(device)
        generated_images = model.decode(z).cpu()
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
    paser.add_argument('--datasets', type=str, nargs='+', default=['Indian_Pines_Corrected', 'KSC_Corrected', 'Pavia', 'PaviaU', 'Salinas_Corrected'], help='which datasets, default all')
    paser.add_argument('--batch_size', type=int, default=20, help='size of the batches')
    paser.add_argument('--image_size', type=int, default=32, help='size of the image')
    paser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    paser.add_argument('--warmup_epoches', type=int, default=0, help='number of warmup epochs of training')
    paser.add_argument('--sample_times', type=int, default=8, help='number of sample times')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/VAE/Salinas_Corrected', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/VAE/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            model = torch.load(torch.load(f"{checkpoint_dir}/model.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            H = W = args.image_size
            in_channels = get_dim(name)
            model = ConvVAE(in_channels, H, W)
            dataloader = get_dataloader(name, args.batch_size, args.image_size)
            train(model, dataloader, num_epochs=args.epochs)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

        sample(model, name, sample_times=args.sample_times, save_full=args.save_full, save_RGB=args.save_RGB, save_dic=args.image_dir)

    print("finish all datasets")
