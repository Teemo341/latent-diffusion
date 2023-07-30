#! /bin/bash
#SBATCH -J KSC_Corrected
#SBATCH -o ./results/ddpm_KSC_Corrected.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:P40:2

python -u main_HSI.py --base configs/HSI-VCA/KSC_Corrected-ldm-VCA.yaml -t --gpus 0,1