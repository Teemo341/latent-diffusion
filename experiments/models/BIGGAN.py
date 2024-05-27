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


class DimReducer(nn.Module):
    def __init__(self):
        super(DimReducer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
    def forward(self, x):
        return self.network(x)
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

class DownBlock(nn.Module): # only downsample the spectral dimension
    def __init__(self, in_channels, out_channels=None):
        super(DownBlock, self).__init__()
        out_channels = out_channels or in_channels//2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        return x

class UpBlock(nn.Module): # only upsample the spectral dimension
    def __init__(self, out_channels, in_channels=None):
        super(UpBlock, self).__init__()
        in_channels = in_channels or out_channels//2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
 
    def forward(self, x):
        x = self.conv_relu(x)
        return x

class DownBlock_2(nn.Module): # only downsample the spectral dimension
    def __init__(self, in_channels, out_channels=None):
        super(DownBlock_2, self).__init__()
        out_channels = out_channels or in_channels//4
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.InstanceNorm2d(out_channels)
 
    def forward(self, x):
        x = self.conv_relu(x)
        x = self.bn(x)
        return x
    
class Encoder_2(nn.Module):
    def __init__(self, dim, hidden_dim=1):
        super(Encoder_2, self).__init__()
        self.down = nn.ModuleList()
        i = 0
        while dim//2 > hidden_dim:
            self.down.append(DownBlock(dim))
            dim = dim // 2
            i = i+1
            # print(i, dim)
        self.down.append(DownBlock(dim,hidden_dim))
        self.resblock1 = ResBlock(hidden_dim)
    def forward(self, x):
        for i in range(len(self.down)):
            layer = self.down[i]
            x = layer(x)
        x = self.resblock1(x)
        return x
    
class Decoder_2(nn.Module):
    def __init__(self, dim, hidden_dim=1):
        super(Decoder_2, self).__init__()
        self.resblock1 = ResBlock(hidden_dim)
        self.up = nn.ModuleList()
        i = 0
        while dim//2 > hidden_dim:
            self.up.append(ResBlock(dim))
            self.up.append(UpBlock(dim))
            dim = dim // 2
            i = i+1
        self.up.append(ResBlock(dim))
        self.up.append(UpBlock(dim,hidden_dim))

    def forward(self, x):
        x = self.resblock1(x)
        for i in reversed(range(len(self.up))):
            layer = self.up[i]
            x = layer(x)
        return x
    
#定义中间层结构
class Midlayer(nn.Module)：
    def __in
# 定义生成器的网络结构
class Generator(nn.Module):
    def __init__(self, latnet_dim, hidden_dim=1):
        super(Generator, self).__init__()
        self.linear = DimReducer()
        self.decoder = Decoder_2(latnet_dim,hidden_dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.linear(x)
        x = F.relu(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.decoder(x)
        return x

# 定义判别器的网络结构
class Discriminator(nn.Module):
    def __init__(self,latnet_dim, hidden_dim=1, h =32, w =32):
        super(Discriminator, self).__init__()
        self.encoder = Encoder_2(latnet_dim,hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*h*w, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.classifier(x)
        return x

# 定义训练函数
def train(generator, discriminator, train_loader, valid_loader = None, num_epochs=100, warmup_epoches = 8, erly_stop = 5):

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-6, betas=(0.5, 0.999))  # 生成器的优化器
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
