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

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(200, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * (H // 8) * (W // 8), 256)
        self.fc21 = nn.Linear(256, 20)  # 均值向量
        self.fc22 = nn.Linear(256, 20)  # 方差向量

        # 解码器部分
        self.decoder_fc = nn.Linear(20, 32 * (H // 8) * (W // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 200, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 32 * (32 // 8) * (32 // 8))
        h = F.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        h = h.view(-1, 32, H // 8, W // 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



# 定义训练函数

def train(generator, discriminator, train_loader, valid_loader = None, num_epochs=100, warmup_epoches = 8, erly_stop = 5):

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))  # 生成器的优化器
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))  # 判别器的优化器
    #scheduler = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1) #学习率的调节器


    # early stop if d_loss and g_loss do not change for erly_stop times, stop training
    d_loss_list = g_loss_list = []
    for i in range(erly_stop):
        d_loss_list.append(torch.tensor(float(i)))
        g_loss_list.append(torch.tensor(float(i)))

    # 开始训练
    for epoch in range(num_epochs):

        #if epoch < warmup_epoches:
        #    discriminator_train_time = 1
        #    print(f"warmup epoches, repeat 5 times for discriminator {epoch+1}/{warmup_epoches}")
        #else:
        #    discriminator_train_time = 1


        discriminator_train_time = 1
        d_step = 1
        g_step = 4


        for i, input_dic in enumerate(train_loader):
            real_images = input_dic["image"]
            # real_images = torch.tensor(real_images)
            real_images = rearrange(real_images, 'b h w c -> b c h w')
            batch_size = real_images.size(0)

            z = torch.randn(real_images.shape).to(device) # 生成假图像
            fake_images = generator(z)

            # 训练判别器
            for _ in range(d_step):
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                real_outputs = discriminator(real_images.to(device)) # 计算真实图像的输出
                real_loss = criterion(real_outputs, torch.ones_like(real_outputs).to(device)) # real_label 1

                z = torch.randn(real_images.shape).to(device) # 生成假图像
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images.detach()) # 计算假图像的输出
                fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs).to(device)) #fake_label 0

                d_loss = real_loss + fake_loss
                d_loss.backward()
                # nn.utils.clip_grad_norm_(parameters=discriminator.parameters(), max_norm=0.01, norm_type=2) # 梯度裁剪，避免梯度爆炸
                optimizer_D.step()
                # scheduler.step()
                
            if epoch > warmup_epoches:
            # 训练生成器
                for _ in range(g_step):
                    optimizer_G.zero_grad()
                    optimizer_D.zero_grad()
                    z = torch.randn(real_images.shape).to(device) # 生成假图像
                    fake_images = generator(z)
                    fake_outputs = discriminator(fake_images)
                    g_loss = -1 * criterion(fake_outputs, torch.zeros_like(fake_outputs).to(device))
                    # print(g_loss)
                    g_loss.backward()
                    nn.utils.clip_grad_norm_(parameters=generator.parameters(), max_norm=0.01, norm_type=2) # 梯度裁剪，避免梯度爆炸
                    optimizer_G.step()

            if "g_loss" in locals():
                if (i+1) % int(len(train_loader)/2) == 0: #一轮打印两次
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

            else:
                if (i+1) % int(len(train_loader)/2) == 0: #一轮打印两次
                     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item():.4f}")        
        # early stop if d_loss and g_loss do not change for erly_stop times, stop training
        #d_loss_list.pop(0)
        #d_loss_list.append(d_loss.item())
        #g_loss_list.pop(0)
        #g_loss_list.append(g_loss.item())
        #if np.var(d_loss_list)/np.mean(d_loss_list) < 0.0001 and np.var(g_loss_list)/np.mean(g_loss_list) < 0.0001:
            #print(f"Early stop at epoch {epoch}")
            #break

        if (epoch+1) % 1 == 0:
            sample(generator, name, sample_times=8, save_full=False, save_RGB=True, h=32, w=32, save_dic=f"./experiments/models/GAN/{epoch+1}")


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
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/GAN', help='directory to save results')
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
