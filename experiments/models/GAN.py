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


# 定义编解码器
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv_bn_relu_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_bn_relu_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        self. conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(dim)
        self.relu_3 = nn.LeakyReLU(inplace=True)
 
    def forward(self, x):
        identity = x
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x += identity
        x = self.relu_3(x)

        return x

class DownBlock(nn.Module): # only downsample the spectral dimension
    def __init__(self, in_channels, out_channels=None):
        super(DownBlock, self).__init__()
        out_channels = out_channels or in_channels//2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
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
            nn.LeakyReLU(inplace=True)
        )
 
    def forward(self, x):
        x = self.conv_relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, hidden_dim=8):
        super(Encoder, self).__init__()
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
    
class Decoder(nn.Module):
    def __init__(self, dim, hidden_dim=8):
        super(Decoder, self).__init__()
        self.resblock1 = ResBlock(hidden_dim)
        self.up = nn.ModuleList()
        i = 0
        while dim//2 > hidden_dim:
            self.up.append(UpBlock(dim))
            dim = dim // 2
            i = i+1
        self.up.append(UpBlock(dim,hidden_dim))

    def forward(self, x):
        x = self.resblock1(x)
        for i in reversed(range(len(self.up))):
            layer = self.up[i]
            # print(i, layer.shape, x.shape)
            x = layer(x)
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
    def __init__(self, dim, hidden_dim=16):
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
    def __init__(self, dim, hidden_dim=16):
        super(Decoder_2, self).__init__()
        self.resblock1 = ResBlock(hidden_dim)
        self.up = nn.ModuleList()
        i = 0
        while dim//2 > hidden_dim:
            self.up.append(UpBlock(dim))
            dim = dim // 2
            i = i+1
        self.up.append(UpBlock(dim,hidden_dim))

    def forward(self, x):
        x = self.resblock1(x)
        for i in reversed(range(len(self.up))):
            layer = self.up[i]
            # print(i, layer.shape, x.shape)
            x = layer(x)
        return x
    

# 定义生成器的网络结构
class Generator(nn.Module):
    def __init__(self, latnet_dim, hidden_dim=16,layers = 9):
        super(Generator, self).__init__()
        self.encoder = Encoder_2(latnet_dim,hidden_dim)
        self.res_block_list = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(layers)])
        self.decoder = Decoder_2(latnet_dim,hidden_dim)
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_block_list(x)
        x = self.decoder(x)
        return x

# 定义判别器的网络结构
class Discriminator(nn.Module):
    def __init__(self,latnet_dim, hidden_dim=16, h =32, w =32):
        super(Discriminator, self).__init__()
        self.encoder = Encoder_2(latnet_dim,hidden_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

# 定义训练函数
def train(generator, discriminator, train_loader, valid_loader = None, num_epochs=100, warmup_epoches = 1, lr = 2e-4, erly_stop = 5):

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))  # 生成器的优化器
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))  # 判别器的优化器
    # optimizer_G = optim.SGD(generator.parameters(), lr=lr)  # 生成器的优化器
    # optimizer_D = optim.SGD(discriminator.parameters(), lr=lr)  # 判别器的优化器
    #scheduler = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1) #学习率的调节器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=50, eta_min=0.00001)

    # early stop if d_loss and g_loss do not change for erly_stop times, stop training
    d_loss_list = g_loss_list = []
    for i in range(erly_stop):
        d_loss_list.append(torch.tensor(float(i)))
        g_loss_list.append(torch.tensor(float(i)))

    # 开始训练
    for epoch in range(num_epochs):

        if epoch < warmup_epoches:
            discriminator_train_time = 1
            generator_train_time = 0
            print(f"warmup epoches {epoch+1}/{warmup_epoches}")
        else:
            discriminator_train_time = 1
            generator_train_time = 1
            discriminator.eval()

        loss_list_d = []
        loss_list_g = []

        for i, input_dic in enumerate(train_loader):
            real_images = input_dic["image"]
            # real_images = torch.tensor(real_images)
            real_images = rearrange(real_images, 'b h w c -> b c h w')
            batch_size = real_images.size(0)
            fake_labels = torch.stack([torch.zeros(batch_size),torch.ones(batch_size)],dim = 1).to(device) # fake_label [0,1]
            real_labels = torch.stack([torch.ones(batch_size),torch.zeros(batch_size)],dim = 1).to(device) # real_label [1,0]

            # 训练判别器
            for _ in range(discriminator_train_time):
                generator.eval()
                discriminator.train()
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                real_outputs = discriminator(real_images.to(device)) # 计算真实图像的输出
                real_loss = criterion(real_outputs, real_labels)

                z = torch.randn(real_images.shape).to(device) # 生成假图像
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images.detach()) # 计算假图像的输出
                fake_loss = criterion(fake_outputs, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                # nn.utils.clip_grad_norm_(parameters=discriminator.parameters(), max_norm=0.01, norm_type=2) # 梯度裁剪，避免梯度爆炸
                optimizer_D.step()
                # scheduler.step()

                loss_list_d.append(d_loss.item()) # record average loss
                
            # 训练生成器
            for _ in range(generator_train_time):
                generator.train()
                discriminator.eval()
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                z = torch.randn(real_images.shape).to(device) # 生成假图像
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images)
                g_loss = 1 * criterion(fake_outputs, fake_labels)
                # print(g_loss)
                g_loss.backward()
                nn.utils.clip_grad_norm_(parameters=generator.parameters(), max_norm=0.001, norm_type=2) # 梯度裁剪，避免梯度爆炸
                optimizer_G.step()

                loss_list_g.append(g_loss.item()) # record average loss

            # peint training information
            if (i+1) % int(len(train_loader)/2) == 0: #一轮打印两次
                if "g_loss" in locals():
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {np.mean(loss_list_d):.4f}, g_loss: {np.mean(loss_list_g):.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {np.mean(loss_list_d):.4f}")       

        # early stop if d_loss and g_loss do not change for erly_stop times, stop training
        #d_loss_list.pop(0)
        #d_loss_list.append(d_loss.item())
        #g_loss_list.pop(0)
        #g_loss_list.append(g_loss.item())
        #if np.var(d_loss_list)/np.mean(d_loss_list) < 0.0001 and np.var(g_loss_list)/np.mean(g_loss_list) < 0.0001:
            #print(f"Early stop at epoch {epoch}")
            #break

        if (epoch+1) % 1 == 0:
            sample(generator, name, sample_times=4, save_full=False, save_RGB=True, h=32, w=32, save_dic=f"./experiments/models/visualization/GAN/{epoch+1}")


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
            generator = Generator(get_dim(name), args.hidden_dim, args.layers)
            discriminator = Discriminator(get_dim(name))
            dataloader = get_dataloader(name, args.batch_size, args.image_size)
            train(generator, discriminator, dataloader, num_epochs=args.epochs, lr = args.lr, warmup_epoches=args.warmup_epoches)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(generator.state_dict(), f"{checkpoint_dir}/generator.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

        sample(generator, name, sample_times=args.sample_times, save_full=args.save_full, save_RGB=args.save_RGB, h=h, w=w, save_dic=args.image_dir)

    print("finish all datasets")
