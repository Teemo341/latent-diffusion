#! /bin/bash
#SBATCH -J Indian_Pines_Corrected
#SBATCH -o ./results/ddpm_Indian_Pines_Corrected.out               
#SBATCH -p compute2
#SBATCH -A compute2
#SBATCH --qos=compute2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:3

python -u main_HSI.py --base configs/HSI-VCA/Indian_Pines_Corrected-ldm-VCA.yaml -t --gpus 0,1,2