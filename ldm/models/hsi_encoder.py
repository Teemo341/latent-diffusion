import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import sys
import torch.linalg as splin
import numpy as np
from einops import rearrange, repeat

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


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

    def encode(self, x):
        # Y = Ae*X
        # Ae.T*Y = Ae.T*Ae*X
        #(Ae.T*Ae)**-1*Ae.T*Y = X
        Y = rearrange(x,'b c h w -> c (b h w)')
        X_ = torch.matmul(torch.linalg.inv(torch.matmul(self.end_members.T,self.end_members)),torch.matmul(self.end_members.T,Y))
        z = rearrange(X_, 'c (b h w) -> b c h w')
        return z


    def decode(self, z):
        X = rearrange(z,'b c h w -> c (b h w)')
        Y = torch.matmul(self.end_members,X)
        dec = rearrange(Y,'c (b h w) -> b c h w')
        return dec
    

    def forward(self, input):
        z = self.encode(input)
        dec = self.decode(z)
        return dec


    def update_endmembers(self, whole_image):
        whole_image = rearrange(whole_image,'c h w -> c (h w)')
        Ae,indice,Yp = vca(whole_image,self.out_channels)
        assert Ae.shape == self.end_members.shape
        self.end_members = Ae.clone()
        self.end_members.requires_grad = True


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
    Y = torch.rand(6,4)
    R =4
    Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    print(Y)
    print(Ae)
    print(indice)
    print(Yp)
    x_ = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(Ae.T,Ae)),Ae.T),Y)
    print(x_)
    print(torch.matmul(Ae,x_))