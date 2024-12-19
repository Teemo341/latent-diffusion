from random import sample
import glob
import torch
from torch.distributions import Categorical
import torch.nn.functional as F_
import numpy as np
import torch.utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from einops import rearrange

from experiments.metric.classifier import *
import ldm.data.HSI as datasets_all


# basic modules

def make_original_HSI(dataset, path = None):
    if path is None:
        path = f'./experiments/metric/original_image/{dataset}/hsi.npy'
    if dataset == 'Indian_Pines_Corrected':
        dataset_ = datasets_all.Indian_Pines_Corrected(augment=False)
        HSI = dataset_.__getitem__(0)['image']  # hwc
    elif dataset == 'KSC_Corrected':
        dataset_ = datasets_all.KSC_Corrected(augment=False)
        HSI = dataset_.__getitem__(0)['image']
    elif dataset == 'Pavia':
        dataset_ = datasets_all.Pavia(augment=False)
        HSI = dataset_.__getitem__(0)['image']
    elif dataset == 'PaviaU':
        dataset_ = datasets_all.PaviaU(augment=False)
        HSI = dataset_.__getitem__(0)['image']
    elif dataset == 'Salinas_Corrected':
        dataset_ = datasets_all.Salinas_Corrected(augment=False)
        HSI = dataset_.__getitem__(0)['image']
    else:
        raise ValueError(f"dataset {dataset} not supported")
    np.save(path, HSI)
    return HSI

def load_original_HSI(dataset, path = None):
    if path is None:
        path = f'./experiments/metric/original_image/{dataset}/hsi.npy'
    HSI = np.load(path)
    return HSI

def load_sampled_HSIs(algorithm, dataset, path = None):
    if algorithm not in ["HUD", "WGANGP", "VAE", "MPRNet"]:
        raise ValueError(f"algorithm {algorithm} not supported")
    if dataset not in ['Indian_Pines_Corrected', 'KSC_Corrected', 'Pavia', 'PaviaU', 'Salinas_Corrected']:
        raise ValueError(f"dataset {dataset} not supported")
    
    sampled_HSI = []
    
    if path is None:
        sample_dir_list = glob.glob(f'./experiments/results/{algorithm}/{dataset}/*.npy')
        for path in sample_dir_list:
            HSI = np.load(path)
            sampled_HSI.append(HSI)
    else:
        HSI = np.load(path)
        sampled_HSI.append(HSI)
    return sampled_HSI

def load_classifier(dataset, path = None):
    if path is None:
        path = f'./experiments/metric/checkpoints/classifier/{dataset}/classifier.pth'
    classifier = Classifier(get_dim(
                dataset), 256, 64, 16, get_label_dim(dataset))
    classifier.load_state_dict(torch.load(path))
    return classifier

# Inception Score
def inception_score(classifier, sampled_images):
    # classifier has softmax
    # all sampled image, has been normed
    classifier.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.to(device)

        # infer predictions
        sampled_images = np.array(sampled_images)
        sampled_images = torch.tensor(sampled_images).to(device)
        sampled_images = rearrange(sampled_images, 'b h w c -> b c h w')
        sampled_predictions = classifier(sampled_images)

        # calculate is
        preds = rearrange(sampled_predictions, 'b c h w -> (b h w) c')
        # preds = torch.tensor([[0.9,0.1],[0.1,0.9],[0.1,0.9]])
        p_yx = preds
        p_y = p_yx.mean(dim = 0)
        KLs = F_.kl_div(p_y.log(), p_yx, reduction='none')
        # KLs = (p_yx * (p_yx / p_y).log()) #? strange, F.kl_div(q.log, p) = p*log(p/q)
        KL = KLs.mean()
        IS = torch.exp(KL).item()
    # print(IS)

    return IS

def Frechet_Inception_Distance(classifier, sampled_images, original_image):
    # classifier has softmax
    # one sampled image, one original image, has been norm
    classifier.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.to(device)
        classifier.classifier = torch.nn.Identity()

        # infer predictions
        original_image = torch.tensor(original_image).to(device) # h w c
        original_image = rearrange(original_image, 'h w c -> 1 c h w')
        sampled_images = np.array(sampled_images)
        sampled_images = torch.tensor(sampled_images).to(device) # b h w c
        sampled_images = rearrange(sampled_images, 'b h w c -> b c h w')
        original_predictions = classifier(original_image)
        sampled_predictions = classifier(sampled_images)
        # print(original_predictions.shape, sampled_predictions.shape)
        original_predictions = rearrange(original_predictions, 'b c h w-> (b h w) c')
        sampled_predictions = rearrange(sampled_predictions, 'b c h w -> (b h w) c')

        # calculate FID
        
        original_predictions = original_predictions.cpu().detach().numpy()
        sampled_predictions = sampled_predictions.cpu().detach().numpy()
        mu1 = np.mean(original_predictions, axis=0)
        mu2 = np.mean(sampled_predictions, axis=0)
        sigma1 = np.cov(original_predictions, rowvar=False)
        sigma2 = np.cov(sampled_predictions, rowvar=False)
        FID = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * linalg.sqrtm(sigma1 @ sigma2))

    return FID

def point_fidelity(sampled_images, original_image):
    # F_p = mean(min(||sampled_pixel-original_pixel||^2))
    F_p = []
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sampled_images = np.array(sampled_images) # b h w c
        sampled_images = torch.tensor(sampled_images,device = device)
        sampled_images = rearrange(sampled_images, 'b h w c -> (b h w) c')
        original_image = torch.tensor(original_image,device = device) # h w c
        for i in range(sampled_images.shape[0]):
            sampled_image = sampled_images[i] # c
            sampled_image = sampled_image.unsqueeze(0).unsqueeze(0) # 1 1 c
            F_p.append((original_image - sampled_image).pow(2).min().item())
        F_p = np.mean(F_p)
    return F_p

def block_diversity(sampled_images, original_image):
    # D_b = mean(min(||sampled_block-original_block||^2))
    D_b = []
    sampled_images = np.array(sampled_images) # b h w c
    original_image = np.array(original_image) # h w c
    slide_h = original_image.shape[0]-sampled_images.shape[1]+1
    slide_w = original_image.shape[1]-sampled_images.shape[2]+1
    for sampled_image in sampled_images:
        D_b_ = []
        for k in range(4): # rotate 4 times, considering that the sampled image may be rotated
            original_image_rotated = np.rot90(original_image, k)
            for i in range(slide_h):
                for j in range(slide_w):
                    original_block = original_image_rotated[i:i+sampled_image.shape[0], j:j+sampled_image.shape[1]]
                    D_b_.append(np.linalg.norm(sampled_image - original_block))
        D_b.append(np.min(D_b_))
    D_b = np.mean(D_b)
    return D_b

def spectral_curve_visualization(HSI,save_path=None):
    # HSI: h w c
    # visualize the spectral curve by hotmap
    # HSI = np.random.random([32,32,200])
    HSI = np.array(HSI)
    HSI = (HSI + 1.0)/2.0
    HSI = rearrange(HSI, 'h w c -> (h w) c')
    x = np.arange(0,HSI.shape[1],1)
    y = np.arange(0,1,0.01)
    C = np.zeros((len(x),len(y)))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]-1):
            C[i,j] = np.sum((y[j]<=HSI[:,x[i]]) * (HSI[:,x[i]]<=y[j+1]))
    
    #plot hotmap
    # C = np.sqrt(C) # augment color
    plt.figure(figsize=(3,3))
    plt.imshow(C.T,aspect='auto',cmap='jet',origin = 'lower')
    plt.yticks([0,25,50,75],[0.0,0.25,0.5,0.75])
    
    if save_path is None:
        save_path = f'./experiments/metric/original_image'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/spectral_curve.png",dpi=100,bbox_inches='tight',pad_inches = 0)
    
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if_make_original_HSI", type=bool, default=False)
    parser.add_argument("--algorithms", type=str, nargs = '+', default=["WGANGP","HUD","VAE","MPRNet"])
    parser.add_argument("--datasets", type=str, nargs= '+', default=["Indian_Pines_Corrected", "KSC_Corrected", "Pavia", "PaviaU", "Salinas_Corrected"])
    parser.add_argument("--metric", type=str, nargs='+', default=["IS", "FID", "F_p", "D_b", "spectral_curve"])
    args = parser.parse_args()


    for dataset in args.datasets:
        print(dataset)
        if args.if_make_original_HSI:
            original_image = make_original_HSI(dataset)
        else:
            original_image = load_original_HSI(dataset)
        classifier = load_classifier(dataset)

        for algorithm in args.algorithms:
            print(algorithm)
            sampled_images = load_sampled_HSIs(algorithm, dataset)

            IS = FID = F_p = D_b = None

            if "IS" in args.metric:
                IS = inception_score(classifier, sampled_images)
                print(f"IS ↑: {IS}")
            if "FID" in args.metric:
                FID = Frechet_Inception_Distance(classifier, sampled_images, original_image)
                print(f"FID↓: {FID}")
            if "F_p" in args.metric:
                F_p = point_fidelity(sampled_images, original_image)
                print(f"F_p↓: {F_p}")
            if "D_b" in args.metric:
                D_b = block_diversity(sampled_images, original_image)
                print(f"D_b↑: {D_b}")
            if "spectral_curve" in args.metric:
                spectral_curve_visualization(original_image, f'./experiments/metric/original_image/{dataset}')
                sampled_images_ = np.array(sampled_images)
                sampled_images_ = rearrange(sampled_images_, 'b h w c -> (b h) w c')
                spectral_curve_visualization(sampled_images_, f'./experiments/metric/{algorithm}/{dataset}')
                print("spectral curve saved")
            
            txt_path = f'./experiments/metric/{algorithm}/{dataset}'
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            with open(f'{txt_path}/metric.txt','w') as f:
                f.write(f"IS↑ : {IS}\n")
                f.write(f"FID↓: {FID}\n")
                f.write(f"F_p↓: {F_p}\n")
                f.write(f"D_b↑: {D_b}\n")

    print("finished")
