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

class VAEmodel(nn.Module):
    def __init__(self):
        super(VAEmodel, self).__init__()

        self.common_fc = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=196),
            nn.Tanh(),
            nn.Linear(in_features=196, out_features=48),
            nn.Tanh(),
        )

        self.mean_fc = nn.Sequential(
            nn.Linear(in_features=48, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        self.log_var_fc = nn.Sequential(
            nn.Linear(in_features=48, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=2)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=48),
            nn.Tanh(),
            nn.Linear(in_features=48, out_features=196),
            nn.Tanh(),
            nn.Linear(in_features=196, out_features=28 * 28),
        )

    def forward(self, x):
        # B, C, W, H
        # Encoding part
        mean, log_var = self.encode(x)
        # Sampling
        z = self.sample(mean, log_var)
        # Decoder part
        out = self.decode(z)
        return mean, log_var, out

    def encode(self, x):
        #print(f'shape of x is {x.shape}')
        out = self.common_fc(torch.flatten(x, start_dim=1))
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var

    def sample(self, mean, log_var):
        # Re-Parameterization Trick z = mean + std * epsilon
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = epsilon * std + mean
        return z

    def decode(self, z):
        out = self.decoder_fc(z)
        out = out.reshape((z.size(0), 1, 28, 28))
        return out