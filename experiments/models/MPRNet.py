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
from torchsummary import summary
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn.functional as F
import math
import warnings

#class MidLayer(nn.Module):
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat,200, kernel_size, bias=bias)
        self.conv3 = conv(200, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class MPRNet(nn.Module):
    def __init__(self, in_c=200, out_c=200, n_feat=200, scale_unetfeats=200, scale_orsnetfeats=200, num_cab=4, kernel_size=3, reduction=1, bias=False):
        super(MPRNet, self).__init__()

        self.conv_in = nn.Conv2d(3, in_c, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                 bias=bias)

        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        b, c, h_inp, w_inp = x3_img.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x3_img = F.pad(x3_img, [0, pad_w, 0, pad_h], mode='reflect')
        x3_img = self.conv_in(x3_img)

        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return (stage3_img + x3_img)[:, :, :h_inp, :w_inp]
    

class HSI_select():
    def __init__(self, name):
        super(HSI_select, self).__init__()
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
        image = HSI_image[:,:,:,self.vis_channels]
        return image   

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion_mrae = Loss_RMSE()
    for epoch in range(num_epochs):
        model.train()
        losses = AverageMeter()
        for i, input_dic in enumerate(train_loader):
            data = input_dic["image"].to(device)
            # real_images = torch.tensor(real_images)
            vis_channels =[46,27,10]
            images = data[:,:,:,vis_channels]
            data = rearrange(data, 'b h w c -> b c h w')
            images = rearrange(images, 'b h w c -> b c h w')
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, data)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            print(f'i {i+1}, MSE Loss: {losses.avg}')
        # 日志输出
        scheduler_cosine.step()

        if (epoch+1) % 5 == 0:
            sample(model, name, sample_times=8, save_full=True, save_RGB=True, save_dic=f"./experiments/models/MPRNet/Indian_Pines_Corrected/{epoch+1}")
# 生成图像

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
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
def sample(model, name, sample_times=8, save_full = True, save_RGB = True, save_dic="./results/MPRNet"):
    save_dic = os.path.join(save_dic, name)
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_data = get_dataloader(name, args.times, args.image_size)
    for i, input_dic in enumerate(test_data):
        if i == 1:
            dataneed = input_dic["image"]    
    dataneed = dataneed.to(device)
    select = HSI_select(name)
    dataneed = select.visualization(dataneed)
    dataneed = rearrange(dataneed, 'b h w c -> b c h w')
    with torch.no_grad():
        generated_images = model(dataneed).cpu()
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
    paser.add_argument('--batch_size', type=int, default=40, help='size of the batches')
    paser.add_argument('--times', type=int, default=8, help='size of the batches')
    paser.add_argument('--image_size', type=int, default=32, help='size of the image')
    paser.add_argument('--epochs', type=int, default=30, help='number of epochs of training')
    paser.add_argument('--warmup_epoches', type=int, default=0, help='number of warmup epochs of training')
    paser.add_argument('--sample_times', type=int, default=8, help='number of sample times')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/models/checkpoints', help='directory to save checkpoints')
    paser.add_argument('--image_dir', type=str, default='./experiments/results/MPRNet/Indian_Pines_Corrected', help='directory to save results')
    paser.add_argument('--save_full', type=bool, default=True, help='save full image or not')
    paser.add_argument('--save_RGB', type=bool, default=True, help='save RGB image or not')

    args = paser.parse_args()
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/MPRNet/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            model = torch.load(torch.load(f"{checkpoint_dir}/model.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            H = W = args.image_size
            in_channels = get_dim(name)
            model = MPRNet(in_c=200, out_c=200, n_feat=200, scale_unetfeats=200, scale_orsnetfeats=200, num_cab=4, kernel_size=3, reduction=1, bias=False)
            dataloader = get_dataloader(name, args.batch_size, args.image_size)

            train(model, dataloader, num_epochs=args.epochs)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")
            print(f"Save checkpoint to {checkpoint_dir}")
        sample(model, name, sample_times=8, save_full=False, save_RGB=True, save_dic=args.image_dir)

    print("finish all datasets")

