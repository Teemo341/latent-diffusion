#! /bin/bash
#SBATCH -J VCA
#SBATCH -o ./results/VCA.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:P40:1

python -m ldm.models.hsi_encoder --data Indian_Pines_Corrected --save_path ./models/first_stage_models/HSI/VCA/ --feature_channels 20