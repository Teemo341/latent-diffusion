import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import sys,os
import torch.linalg as splin
import numpy as np
from einops import rearrange, repeat
import random
import matplotlib.pyplot as plt

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import ldm.data.HSI as datasets_all

from ldm.util import instantiate_from_config
import argparse
from PIL import Image


#############################################
# Internal functions
#############################################

def estimate_snr(Y,r_m,x):

  [L, N] = Y.shape           # L number of bands (channels), N number of pixels
  [p, N] = x.shape           # p number of endmembers (reduced dimension)
  
  P_y     = torch.sum(Y**2)/float(N)
  P_x     = torch.sum(x**2)/float(N) + torch.sum(r_m**2)
  snr_est = 10*torch.log10( (P_x - p/L*P_y)/(P_y - P_x) )

  return snr_est



def vca(Y,R,verbose = True,snr_input = 0):
# Vertex Component Analysis
#
# Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
#
# ------- Input variables -------------
#  Y - matrix with dimensions L(channels) x N(pixels)
#      each pixel is a linear mixture of R endmembers
#      signatures Y = M x s, where s = gamma x alfa
#      gamma is a illumination perturbation factor and
#      alfa are the abundance fractions of each endmember.
#  R - positive integer number of endmembers in the scene
#
# ------- Output variables -----------
# Ae     - estimated mixing matrix (endmembers signatures)
# indice - pixels that were chosen to be the most pure
# Yp     - Data matrix Y projected.   
#
# ------- Optional parameters---------
# snr_input - (float) signal to noise ratio (dB)
# v         - [True | False]
# ------------------------------------
#
# Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
# This code is a translation of a matlab code provided by 
# Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
# available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
# Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
#
# more details on:
# Jose M. P. Nascimento and Jose M. B. Dias 
# "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
# submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
# 
# 

  #############################################
  # Initializations
  #############################################
  Y = torch.tensor(Y)
  if len(Y.shape)!=2:
    sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

  [L, N]=Y.shape   # L number of bands (channels), N number of pixels
       
  R = int(R)
  if (R<0 or R>L):  
    sys.exit('ENDMEMBER parameter must be integer between 1 and L')
        
  #############################################
  # SNR Estimates
  #############################################

  if snr_input==0:
    y_m = torch.mean(Y, dim = 1, keepdim = True)
    Y_o = Y - y_m           # data with zero-mean
    Ud  = splin.svd(torch.matmul(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
    x_p = torch.matmul(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace

    SNR = estimate_snr(Y,y_m,x_p);
    
    if verbose:
      print("SNR estimated = {}[dB]".format(SNR))
  else:
    SNR = snr_input
    if verbose:
      print("input SNR = {}[dB]\n".format(SNR))

  SNR_th = 15 + 10*np.log10(R)
         
  #############################################
  # Choosing Projective Projection or 
  #          projection to p-1 subspace
  #############################################

  if SNR < SNR_th:
    if verbose:
      print("... Select proj. to R-1")
                
      d = R-1
      if snr_input==0: # it means that the projection is already computed
        Ud = Ud[:,:d]
      else:
        y_m = torch.mean(Y,dim=1,keepdim=True)
        Y_o = Y - y_m  # data with zero-mean 
         
        Ud  = splin.svd(torch.matmul(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix 
        x_p =  torch.matmul(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace
                
      Yp =  torch.matmul(Ud,x_p[:d,:]) + y_m      # again in dimension L
                
      x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
      c = torch.amax(torch.sum(x**2,dim=0))**0.5
      y = torch.vstack(( x, c*torch.ones((1,N)) ))
  else:
    if verbose:
      print("... Select the projective proj.")
             
    d = R
    Ud  = splin.svd(torch.matmul(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
                
    x_p = torch.matmul(Ud.T,Y)
    Yp =  torch.matmul(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
                
    x =  torch.matmul(Ud.T,Y)
    u = torch.mean(x,dim=1,keepdim=True)        #equivalent to  u = Ud.T * r_m
    y =  x / torch.matmul(u.T,x)

 
  #############################################
  # VCA algorithm
  #############################################

  indice = np.zeros((R),dtype=int)
  A = torch.zeros((R,R))
  A[-1,0] = 1

  for i in range(R):
    w = torch.rand(R,1);   
    f = w - torch.matmul(A,torch.matmul(splin.pinv(A),w))
    f = f / splin.norm(f)
      
    v = torch.matmul(f.T,y)

    indice[i] = np.argmax(np.absolute(v))
    A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))

  Ae = Yp[:,indice]

  return Ae,indice,Yp


class VCA(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ckpt_path=None
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.end_members = torch.nn.Parameter(torch.randn(in_channels,out_channels))
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    

    def encode_(self,x):
      # https://github.com/isaacgerg/matlabHyperspectralToolbox/blob/master/hyperspectralToolbox/hyperFcls.m
      # %HYPERFCLS Performs fully constrained least squares on pixels of M.
      # % hyperFcls performs fully constrained least squares of each pixel in M 
      # % using the endmember signatures of U.  Fully constrained least squares 
      # % is least squares with the abundance sum-to-one constraint (ASC) and the 
      # % abundance nonnegative constraint (ANC).
      # %
      # % Usage
      # %   [ X ] = hyperFcls( M, U )
      # % Inputs
      # %   M - HSI data matrix (p x N)
      # %   U - Matrix of endmembers (p x q)
      # % Outputs
      # %   X - Abundance maps (q x N)
      # %
      # % References
      # %   "Fully Constrained Least-Squares Based Linear Unmixing." Daniel Heinz, 
      # % Chein-I Chang, and Mark L.G. Althouse. IEEE. 1999.

      b_,c_,h_,w_ = x.shape
      Y = rearrange(x,'b c h w -> c (b h w)',b=b_,c=c_,h=h_,w=w_)
      [p1, N] = Y.shape
      [p2, q] = self.end_members.shape
      p = p1
      X = torch.zeros([q, N]);
      U = self.end_members.clone()
      for n1 in range(N):
          count = q
          done = 0
          ref = torch.arange(0,q)
          r = Y[:, n1]
          U = self.end_members.clone()
          while not(done):
              U_inv = splin.inv(torch.matmul(U.T,U))
              als_hat = torch.matmul(U_inv,torch.matmul(U.T,r))
              s = torch.matmul(U_inv,torch.ones(count, 1)).reshape(-1)
              afcls_hat = als_hat - (U_inv@torch.ones(count, 1)).reshape(-1)*splin.inv(torch.ones(1, count)@U_inv@torch.ones(count, 1))*(torch.ones(1, count)@als_hat-1)
              afcls_hat = afcls_hat.reshape(-1)

              # % See if all components are positive.  If so, then stop.
              if (torch.sum(afcls_hat>0) == count):
                  alpha = torch.zeros(q)
                  alpha[ref] = afcls_hat
                  break;
              # % Multiply negative elements by their counterpart in the s vector.
              # % Find largest abs(a_ij, s_ij) and remove entry from alpha.
              idx = torch.where(afcls_hat<0)[0]
              afcls_hat[idx] = afcls_hat[idx]/s[idx]
              [val, maxIdx] = torch.max(torch.abs(afcls_hat[idx]),dim=0)
              maxIdx = idx[maxIdx]
              # alpha[maxIdx] = 0 #? not sure
              keep = torch.tensor(list(set(range(U.shape[1])) - {int(maxIdx)}), dtype=torch.long)
              U = U[:, keep]
              count = count - 1;
              ref = ref[keep]
          X[:, n1] = alpha.squeeze()

      c_ = self.out_channels
      z = rearrange(X, 'c (b h w) -> b c h w',b=b_,c=c_,h=h_,w=w_)
      z = torch.clamp(z,-1,1)
      return z

    def encode(self, x):
        # Y = Ae*X
        # Ae.T*Y = Ae.T*Ae*X
        #(Ae.T*Ae)**-1*Ae.T*Y = X
        
        x = x+1.0 # for unmixing, all members should be >0, so [-1,1]->[0,2]
        b_,c_,h_,w_ = x.shape
        Y = rearrange(x,'b c h w -> c (b h w)',b=b_,c=c_,h=h_,w=w_)
        X_ = torch.matmul(torch.linalg.inv(torch.matmul(self.end_members.T,self.end_members)),torch.matmul(self.end_members.T,Y))
        c_ = self.out_channels
        z = rearrange(X_, 'c (b h w) -> b c h w',b=b_,c=c_,h=h_,w=w_)
        z = z*2.0-1.0 # for unmixing, z should be about [0,1], so [0,1]->[-1,1]
        return z


    def decode(self, z):
        z = (z+1.0)/2.0 # for unmixing, z should be about [0,1]
        b_,c_,h_,w_ = z.shape
        X = rearrange(z,'b c h w -> c (b h w)',b=b_,c=c_,h=h_,w=w_)
        Y = torch.matmul(self.end_members,X)
        c_ = self.in_channels
        dec = rearrange(Y,'c (b h w) -> b c h w',b=b_,c=c_,h=h_,w=w_)
        dec = dec-1.0 # for unmixing, return to [-1,1]
        return dec
    

    def forward(self, input):
        z = self.encode(input)
        dec = self.decode(z)
        return dec
    

    def get_input(self, batch, k='image'):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        if not only_inputs:
            xrec = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


def update_endmembers(VCA_module, whole_image):
    whole_image = whole_image+1.0 # for unmixing, all members should be >0, so [-1,1]->[0,2]
    
    # whole_image = rearrange(whole_image,'c h w -> c (h w)')
    whole_image = rearrange(whole_image,'h w c -> c (h w)')
    Ae,indice,Yp = vca(whole_image,VCA_module.out_channels)
    # print(Ae.shape)
    # print(VCA_module.end_members.shape)
    assert Ae.shape == VCA_module.end_members.shape
    VCA_module.end_members.data = Ae.clone()
    VCA_module.end_members.requires_grad = True

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data', type=str,help='allpy vca on which HSI data',choices=['Indian_Pines_Corrected','KSC_Corrected','Pavia','PaviaU','Salinas_Corrected'])
    parser.add_argument('--save_path',type = str,default="./models/first_stage_models/HSI/VCA/")
    parser.add_argument('--feature_channels',type = int,default=20)
    parser.add_argument('--visual_channels',type = int,default=[28,9,10],nargs='+')
    parser.add_argument('--seed',type = int,default=0)
    args = parser.parse_args()

    if args.seed:
      random.seed(args.seed)

    if args.data == 'Indian_Pines_Corrected':
       dataset = datasets_all.Indian_Pines_Corrected()
    elif args.data == 'KSC_Corrected':
       dataset = datasets_all.KSC_Corrected()
    elif args.data == 'Pavia':
       dataset = datasets_all.Pavia()
    elif args.data == 'PaviaU':
       dataset = datasets_all.PaviaU()
    elif args.data == 'Salinas_Corrected':
       dataset = datasets_all.Salinas_Corrected()


    print('max value:',dataset.get_max())
    image = dataset.__getitem__(0)['image']#hwc

    h,w,c = image.shape
    model = VCA(in_channels=c,out_channels=args.feature_channels)
    update_endmembers(model,image)

    if not os.path.exists(args.save_path+args.data):
       os.makedirs(args.save_path+args.data)
    path = args.save_path+args.data+'/model.ckpt' 
    stat = {}
    stat['state_dict']=model.state_dict()
    torch.save(stat,path)

    # show inputs and reconstructions
    b = (image+1.0)/2.0
    b = b[:,:,args.visual_channels]
    b = (b*255).astype(np.uint8)
    print('input',b.shape)
    b = Image.fromarray(b)
    b.save(args.save_path+args.data+"/original_image.png")

    b = image
    b = torch.tensor(b)
    b = b[None,...]# bhwc
    b = rearrange(b,'b h w c -> b c h w')
    z = model.encode(b)
    print('abundance',z.shape)

    z_ = rearrange(z, 'b c h w -> b h w c')
    z_ = z_[0,:,:,:]
    z_ = z_.reshape(-1)
    plt.figure(1)
    plt.hist(z_.detach().numpy(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.savefig(args.save_path+args.data+"/hist_image.png")
    z_ = rearrange(z, 'b c h w -> b h w c')
    z_ = z_[0,:,:,:]
    print('z value range',z_.min(),z_.max(),z_.abs().min())
    z_ = np.array(((z_-z_.min())/(z_.max()-z_.min())).detach()*225)
    z_ = z_[:,:,[0,1,2]].astype(np.uint8)
    z_ = Image.fromarray(z_)
    z_.save(args.save_path+args.data+"/abundance_image.png")

    b_ = model.decode(z)
    b_ = rearrange(b_, 'b c h w -> b h w c')
    b_ = b_[0,:,:,:]
    b_ = np.array((b_.detach()+1.0)/2*225)
    b_ = b_[:,:,args.visual_channels].astype(np.uint8)
    print('resonstruct',b_.shape)
    b_ = Image.fromarray(b_)
    b_.save(args.save_path+args.data+"/reconstructed_image.png")

    z_ = z
    # z_ = z_*0.5233 + torch.randn_like(z_)*0.8521
    z_ = torch.randn_like(z_)
    b_ = model.decode(z_)
    b_ = rearrange(b_, 'b c h w -> b h w c')
    b_ = b_[0,:,:,:]
    b_ = np.array((b_.detach()+1.0)/2*225)
    b_ = b_[:,:,args.visual_channels].astype(np.uint8)
    b_ = Image.fromarray(b_)
    b_.save(args.save_path+args.data+"/reconstructed_noise_image.png")

    

    print('finished')

    # model = VCA(in_channels=200,out_channels=20,ckpt_path='models/first_stage_models/HSI/VCA/model.ckpt')
    # print('finished')