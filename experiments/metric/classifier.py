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


class Classifier(nn.Module):
    def __init__(self,latnet_dim, embedding_dim = 256, hidden_dim=64, layers=1, classifier_dim= 17):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(latnet_dim, embedding_dim,1), #1x1conv
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU()
        )
        self.Conv2 = nn.Sequential(*[nn.Sequential(nn.Conv2d(embedding_dim, embedding_dim,1), nn.BatchNorm2d(embedding_dim), nn.LeakyReLU()) for _ in range(layers)])
        self.dropout = nn.Dropout(0.5)
        i = embedding_dim
        self.Conv3 = nn.Sequential()
        while i > hidden_dim:
            self.Conv3.append(nn.Sequential(
                nn.Conv2d(i, i//2,1),
                nn.BatchNorm2d(i//2),
                nn.LeakyReLU()
            ))
            i = i//2
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, classifier_dim,1),
            nn.Softmax2d()
        )

    def forward(self, x):
        x = self.Conv1(x)
        # x = self.dropout(x)
        x = self.Conv2(x)
        x = self.dropout(x)
        x = self.Conv3(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# 定义训练函数
def train(classifier, train_loader, valid_loader = None, num_epochs=100, lr = 2e-4, erly_stop = 5):

    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    # optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler_step = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    scheduler_epoch = optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)


    # early stop if loss do not change for erly_stop times, stop training
    loss_list_epoch = []
    for i in range(erly_stop):
        loss_list_epoch.append(torch.tensor(float(i)))

    # 开始训练
    for epoch in range(num_epochs):
        loss_list_train = []
        acc_list_train = []
        for i, input_dic in enumerate(train_loader):
            images = input_dic["image"]
            images = rearrange(images, 'b h w c -> b c h w')
            images = images.to(device)
            labels = input_dic["label"]
            labels = rearrange(labels, 'b h w c -> b c h w')
            labels = labels.to(device)

            # 训练
            classifier.train()
            optimizer.zero_grad()
            prediction = classifier(images)
            loss = criterion(prediction, labels) # real_label 1
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=discriminator.parameters(), max_norm=0.01, norm_type=2) # 梯度裁剪，避免梯度爆炸
            optimizer.step()

            # record average loss
            prediction = prediction.argmax(dim=1)
            acc = (prediction == labels.argmax(dim=1)).sum().item() / (images.shape[0] * images.shape[2] * images.shape[3])
            loss_list_train.append(loss.item())
            acc_list_train.append(acc)

            if (i+1) % int(len(train_loader)/1) == 0: #一轮验证1次,或者打印1次训练
                if valid_loader is not None:
                    loss_list_valid = []
                    acc_list_valid = []
                    classifier.eval()
                    for j, input_dic_ in enumerate(valid_loader):
                        images_ = input_dic_["image"]
                        images_ = rearrange(images_, 'b h w c -> b c h w')
                        images_ = images_.to(device)
                        labels_ = input_dic_["label"]
                        labels_ = rearrange(labels_, 'b h w c -> b c h w')
                        labels_ = labels_.to(device)

                        prediction_ = classifier(images_)
                        loss_ = criterion(prediction_, labels_)
                        prediction_ = prediction_.argmax(dim=1)
                        acc = (prediction_ == labels_.argmax(dim=1)).sum().item() / (images_.shape[0] * images_.shape[2] * images_.shape[3])

                        loss_list_valid.append(loss_.item())
                        acc_list_valid.append(acc)

                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], train_loss: {np.mean(loss_list_train):.4f}, train_acc: {np.mean(acc_list_train):.4f}, valid_loss: {np.mean(loss_list_valid):.4f}, valid_acc: {np.mean(acc_list_valid):.4f}")

                    # record early stop loss
                    loss_list_epoch.pop(0)
                    loss_list_epoch.append(np.mean(loss_list_valid))

                else:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], train_loss: {np.mean(loss_list_train):.4f}, train_acc: {np.mean(acc_list_train):.4f}")

                    # record early stop loss
                    loss_list_epoch.pop(0)
                    loss_list_epoch.append(np.mean(loss_list_train))

            # scheduler_step.step()
        # scheduler_epoch.step()  
                  
        # early stop if loss do not change for erly_stop times, stop training
        if np.var(loss_list_epoch)/np.mean(loss_list_epoch) < 1e-8:
            print(f"Early stop at epoch {epoch}")
            break


# 加载数据集
def get_train_loader(name,batch_size,image_size): # b h w c
    if name == 'Indian_Pines_Corrected':
        dataset = Indian_Pines_Corrected_train(size = image_size)
        #(145, 145, 200).[36,17,11]
    elif name == 'KSC_Corrected':
        dataset = KSC_Corrected_train(size = image_size)
        #(512, 614, 176).[28,9,10]
    elif name == "Pavia":
        dataset = Pavia_train(size = image_size)
        #((1093, 715, 102).[46,27,10]
    elif name == "PaviaU":
        dataset = PaviaU_train(size = image_size)
        #(610, 340, 103).[46,27,10]
    elif name == "Salinas_Corrected":
        dataset = Salinas_Corrected_train(size = image_size)
        #(512, 217, 204).[36,17,11]
    else:
        raise ValueError("Unsupported dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    return dataloader

def get_valid_loader(name,batch_size,image_size): # b h w c
    if name == 'Indian_Pines_Corrected':
        dataset = Indian_Pines_Corrected_valid(size = image_size)
        #(145, 145, 200).[36,17,11]
    elif name == 'KSC_Corrected':
        dataset = KSC_Corrected_valid(size = image_size)
        #(512, 614, 176).[28,9,10]
    elif name == "Pavia":
        dataset = Pavia_valid(size = image_size)
        #((1093, 715, 102).[46,27,10]
    elif name == "PaviaU":
        dataset = PaviaU_valid(size = image_size)
        #(610, 340, 103).[46,27,10]
    elif name == "Salinas_Corrected":
        dataset = Salinas_Corrected_valid(size = image_size)
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

def get_label_dim(name):
    dic = {}
    dic["Indian_Pines_Corrected"] = 17
    dic["KSC_Corrected"] = 14
    dic["Pavia"] = 10
    dic["PaviaU"] = 10
    dic["Salinas_Corrected"] = 17
    return dic[name]


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    # dataset
    paser.add_argument('--datasets', type=str, nargs='+', default=['Indian_Pines_Corrected', 'KSC_Corrected', 'Pavia', 'PaviaU', 'Salinas_Corrected'], help='which datasets, default all')
    paser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    paser.add_argument('--image_size', type=int, default=32, help='size of the image')
    # model hyperparam
    paser.add_argument('--embedding_dim', type=int, default=256, help='dimensionality of the embedding space')
    paser.add_argument('--hidden_dim', type=int, default=64, help='dimensionality of the hidden space')
    paser.add_argument('--layers', type=int, default=1, help='number of layers')
    # training hyperparam
    paser.add_argument('--epochs', type=int, default=10, help='number of epochs of training')
    paser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    paser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint or not')
    paser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint or not')
    paser.add_argument('--checkpoint_dir', type=str, default='./experiments/metric/checkpoints', help='directory to save checkpoints')

    args = paser.parse_args()

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))
    
    for name in args.datasets:
        if args.save_checkpoint or args.load_checkpoint:
            checkpoint_dir = f"{args.checkpoint_dir}/classifier/{name}"           
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if args.load_checkpoint:
            classifier = torch.load(torch.load(f"{checkpoint_dir}/classifier.pth"))
            print(f"Load checkpoint from {checkpoint_dir}")
        else:
            print(f"Start training {name} dataset")

            classifier = Classifier(get_dim(name), args.embedding_dim, args.hidden_dim,args.layers, get_label_dim(name))
            train_loader = get_train_loader(name, args.batch_size, args.image_size)
            valid_loader = get_valid_loader(name, args.batch_size, args.image_size)
            train(classifier, train_loader, valid_loader, num_epochs=args.epochs, lr = args.lr)

            print(f"Finish training {name} dataset")

        if args.save_checkpoint:
            torch.save(classifier.state_dict(), f"{checkpoint_dir}/classifier.pth")
            print(f"Save checkpoint to {checkpoint_dir}")

    print("finish all datasets")
