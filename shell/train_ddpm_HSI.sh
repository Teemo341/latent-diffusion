#! /bin/bash
#SBATCH -J PaviaU
#SBATCH -o ./results/ddpm_PaviaU.out               
#SBATCH -p compute1
#SBATCH -A compute1
#SBATCH --qos=compute1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

python -u main_HSI.py --logdir logs/PaviaU --base configs/HSI-VCA/PaviaU-ldm-VCA.yaml -t --gpus 0,1