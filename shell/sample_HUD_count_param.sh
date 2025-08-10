#! /bin/bash
#SBATCH -J paramcount_HUD
#SBATCH -o ./results/paramcount_HUD.out               
#SBATCH -p compute1    
#SBATCH -A compute1              
#SBATCH --qos=compute1              
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4090:1

python -u -m scripts.param_count_HSI \
    --dataset Indian_Pines_Corrected \
    --resume ./models/HUD \
    --n_samples 4 \
    --eta 1.0 \
    --logdir ./experiments/results/HUD  \
    --custom_steps 250 \
    --batch_size 4

    