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

class Generator(torch.nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = torch.nn.Sequential(
            # 第一层: 从噪声生成 4x4 特征图
            self._block(in_channels=channels_noise, out_channels=features_g * 8, kernel_size=4, stride=1, padding=0),  # 输出: (b, features_g * 8, 4, 4)

            # 插入层: 4x4 -> 4x4 (不改变尺寸，只调整通道数)
            self._block(in_channels=features_g * 8, out_channels=features_g * 8, kernel_size=3, stride=1, padding=1),  # 输出: (b, features_g * 8, 4, 4)

            # 第二层: 4x4 -> 8x8
            self._block(in_channels=features_g * 8, out_channels=features_g * 4, kernel_size=4, stride=2, padding=1),  # 输出: (b, features_g * 4, 8, 8)

            # 插入层: 8x8 -> 8x8 (不改变尺寸，只调整通道数)
            self._block(in_channels=features_g * 4, out_channels=features_g * 4, kernel_size=3, stride=1, padding=1),  # 输出: (b, features_g * 4, 8, 8)

            # 第三层: 8x8 -> 16x16
            self._block(in_channels=features_g * 4, out_channels=features_g * 2, kernel_size=4, stride=2, padding=1),  # 输出: (b, features_g * 2, 16, 16)

            # 插入层: 16x16 -> 16x16 (不改变尺寸，只调整通道数)
            self._block(in_channels=features_g * 2, out_channels=features_g * 2, kernel_size=3, stride=1, padding=1),  # 输出: (b, features_g * 2, 16, 16)

            # 第四层: 16x16 -> 32x32
            torch.nn.ConvTranspose2d(
                in_channels=features_g * 2, out_channels=channels_img, kernel_size=4, stride=2, padding=1  # 输出: (b, channels_img, 32, 32)
            ),
            torch.nn.Tanh()
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU()
        )
        return self.conv

    def forward(self,input):
        x = self.gen(input)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator, self).__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channels_img,out_channels=features_d,kernel_size=(4,4),stride=(2,2),padding=(1,1)
            ),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True),
            self._block(in_channels=features_d,out_channels=features_d * 2,kernel_size=(4,4),stride=(2,2),
                        padding=(1,1)),
            self._block(in_channels=features_d * 2, out_channels=features_d * 4, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            torch.nn.Conv2d(in_channels=features_d*4,out_channels=1,kernel_size=(4,4),stride=(2,2),padding=0)
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            #affine=True:一个布尔值，当设置为True时，该模块具有可学习的仿射参数，以与批量规范化相同的方式初始化。默认值：False。
            torch.nn.InstanceNorm2d(num_features=out_channels,affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )
        return self.conv
    def forward(self,input):
        x = self.disc(input)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.BatchNorm2d)):
            torch.nn.init.normal(m.weight.data,0.0,0.02)

def gradient_penality(critic,real,fake):
    """
    :param critic: 判别器模型
    :param real: 真实样本
    :param fake: 生成的样本
    :param device: 设备CUP or GPU
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE,C,H,W = real.shape
    alpha = torch.randn(size=(BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images=real*alpha + fake*(1-alpha)

    #计算判别器输出
    mixed_scores = critic(interpolated_images)
    #求导
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.reshape(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penality = torch.mean((gradient_norm - 1)**2)
    return gradient_penality

def train(gen, critic, train_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen.to(device)
    critic.to(device)
    #定义相关参数
    LEARNING_RATE = 1e-4#5e-5
    CRITIC_ITERATIONS = 5
    LAMBDA_GP=10
    Z_DIM = 100
    initialize_weights(gen)
    initialize_weights(critic)

    #定义优化器
    opt_gen = torch.optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
    opt_critic = torch.optim.Adam(critic.parameters(),lr=0.00005,betas=(0.0,0.9))

#定义随机噪声
    fixed_noise = torch.randn(size = (16,Z_DIM,1,1),device=device)
    gen.train()
    critic.train()

    step = 0
    for epoch in range(num_epochs):
        for batch_idx, input_dic in enumerate(train_loader):
            data = input_dic["image"]
            # real_images = torch.tensor(real_images)
            data = rearrange(data, 'b h w c -> b c h w')
            data = data.to(device)
            cur_batch_size = data.shape[0]
            #Train: Critic : max[critic(real)] - E[critic(fake)]
            loss_critic = 0
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(size = (cur_batch_size,Z_DIM,1,1),device=device)
                fake_img = gen(noise)
                #使用reshape主要是将最后的维度从[1,1,1,1]=>[1]
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake_img).reshape(-1)

                gp = gradient_penality(critic,data,fake_img)

                loss_critic = -(torch.mean(critic_real)- torch.mean(critic_fake)) + LAMBDA_GP*gp
                opt_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            #将维度从[1,1,1,1]=>[1]
            gen_fake = critic(fake_img).reshape(-1)
            #max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            loss_gen = -torch.mean(gen_fake)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0 and batch_idx > 0:
                gen.eval()
                critic.eval()
                print(
                    f"Epoch[{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)}\
                    Loss D: {loss_critic:.6f},Loss G: {loss_gen:.6f}"
                )
                # with torch.no_grad():
                #     fake_img = gen(fixed_noise)
                #     DrawGen(gen,epoch,fixed_noise)
                #     img_grid_real = torchvision.utils.make_grid(
                #         data,normalize=True
                #     )
                #     img_grid_fake = torchvision.utils.make_grid(
                #         fake_img,normalize=True
                #     )
                #     writer_real.add_image("RealImg",img_grid_real,global_step=step)
                #     writer_fake.add_image("fakeImg",img_grid_fake,global_step=step)
                step += 1
                gen.train()
                critic.train()
        if (epoch+1) % 5 == 0:
                sample(gen, name, sample_times=8, save_full=True, save_RGB=True, h=32, w=32, save_dic=f"./experiments/models/WGAN/Salinas_Corrected/{epoch+1}")

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
        z = torch.randn(size = (sample_times,100,1,1),device=device)
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
    paser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
    paser.add_argument('--warmup_epoches', type=int, default=0, help='number of warmup epochs of training')
    paser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    paser.add_argument('--sample_times', type=int, default=8, help='number of sample times')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/WGAN/Salinas_Corrected', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/GAN/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            Z_DIM = 100
            FEATURES_DISC = 32
            FEATURES_GEN = 16
            CHANNELS_IMG = 204
            #实例模型
            gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN)
            #这里的判别模型使用cirtic，主要是区别于之前的discriminator
            critic = Discriminator(CHANNELS_IMG,FEATURES_DISC)
            gen.load_state_dict(torch.load(f"{checkpoint_dir}/generator.pth"))
            critic.load_state_dict(torch.load(f"{checkpoint_dir}/discriminator.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            h = w = args.image_size
            Z_DIM = 100
            FEATURES_DISC = 32
            FEATURES_GEN = 16
            CHANNELS_IMG = 204
            #实例模型
            gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN)
            #这里的判别模型使用cirtic，主要是区别于之前的discriminator
            critic = Discriminator(CHANNELS_IMG,FEATURES_DISC)
            dataloader = get_dataloader(name, args.batch_size, args.image_size)
            train(gen, critic, dataloader, num_epochs=args.epochs)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(gen.state_dict(), f"{checkpoint_dir}/generator.pth")
            torch.save(critic.state_dict(), f"{checkpoint_dir}/discriminator.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

        sample(gen, name, sample_times=args.sample_times, save_full=args.save_full, save_RGB=args.save_RGB, h=h, w=w, save_dic=args.image_dir)

    print("finish all datasets")

          

