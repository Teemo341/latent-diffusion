#! /bin/bash
#SBATCH -J VCA
#SBATCH -o ./results/VCA.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m ldm.models.hsi_encoder  --data PaviaU --save_path ./models/first_stage_models/HSI/VCA/ --feature_channels 5 --seed 0 --visual_channels 46 27 10