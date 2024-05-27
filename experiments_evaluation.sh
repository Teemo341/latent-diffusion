#! /bin/bash
#SBATCH -J evaluation
#SBATCH -o ./results/experiments_evaluation.out               
#SBATCH -p compute1                 
#SBATCH --qos=normal               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m experiments.metric.evaluation --algorithm GAN --dataset Indian_Pines_Corrected --metric spectral_curve