#! /bin/bash
#SBATCH -J GAN1
#SBATCH -o ./results/GAN1.out               
#SBATCH -p compute1                 
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m experiments.models.BIGGAN  --datasets Indian_Pines_Corrected 