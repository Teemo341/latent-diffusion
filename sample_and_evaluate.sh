#! /bin/bash
#SBATCH -J evaluation
#SBATCH -o ./results/experiments_evaluation_HUD_Pavia.out              
#SBATCH -p compute1    
#SBATCH -A compute1              
#SBATCH --qos=compute1              
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -w node8

python -u -m scripts.sample_diffusion_HSI \
    --dataset Pavia \
    --resume ./models/HUD \
    --n_samples 8 \
    --eta 1.0 \
    --logdir ./experiments/results/HUD  \
    --custom_steps 250 \
    --batch_size 4

python -u -m experiments.metric.evaluation --algorithms HUD --datasets Pavia --metric spectral_curve --if_make_original_HSI False 