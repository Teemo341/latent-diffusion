#! /bin/bash
#SBATCH -J ddpm_lsun
#SBATCH -o ./result/ddpm_lsun.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3090:1

python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml -t --gpus 1