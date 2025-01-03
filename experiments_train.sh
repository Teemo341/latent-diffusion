#! /bin/bash
#SBATCH -J GAN
#SBATCH -o ./results/GAN.out               
#SBATCH -p compute1             
#SBATCH -A compute1    
#SBATCH --qos=compute1               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m experiments.models.GAN  --datasets Indian_Pines_Corrected --hidden_dim 32 --layers 8 --epochs 30 --warmup_epoches 3 --lr 2e-5