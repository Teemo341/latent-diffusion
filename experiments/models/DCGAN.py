# 导入torch模块
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ldm.data.HSI import *
import numpy as np
from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 200, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(200, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function


def train(generator, discriminator, train_loader, num_epochs=10):
    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adversarial_loss = torch.nn.BCELoss()
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

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

                    # Adversarial ground truths
            valid = Variable(torch.tensor(real_images.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
            fake = Variable(torch.tensor(real_images.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

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

for epoch in range(num_epochs):
    for i, input_dic in enumerate(train_loader):
        real_images = input_dic["image"]
        # real_images = torch.tensor(real_images)
        real_images = rearrange(real_images, 'b h w c -> b c h w')
        real_images = real_images.to(device)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)