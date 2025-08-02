#! /bin/bash
#SBATCH -J ddpm_lsun
#SBATCH -o ./results/ddpm_lsun.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:P40:1

python -u main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml -t 