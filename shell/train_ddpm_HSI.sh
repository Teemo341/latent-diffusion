#! /bin/bash
#SBATCH -J Salinas_Corrected
#SBATCH -o ./results/ddpm_Salinas_Corrected.out               
#SBATCH -p compute1
#SBATCH -A compute1
#SBATCH --qos=compute1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

python -u main_HSI.py --logdir logs/Salinas_Corrected --base configs/HSI-VCA/Salinas_Corrected-ldm-VCA.yaml -t --gpus 1,