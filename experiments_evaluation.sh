#! /bin/bash
#SBATCH -J evaluation
#SBATCH -o ./results/experiments_evaluation_HUD.out
#SBATCH -p compute1
#SBATCH -A compute1
#SBATCH --qos=compute1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -w node8

python -u -m experiments.metric.evaluation --algorithms HUD --datasets Indian_Pines_Corrected --if_make_original_HSI False