#! /bin/bash
#SBATCH -J PaviaU
#SBATCH -o ./results/ddpm_PaviaU.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:3090:1

python -u main_HSI.py --base configs/HSI-VCA/PaviaU-ldm-VCA.yaml -t --gpus 0,