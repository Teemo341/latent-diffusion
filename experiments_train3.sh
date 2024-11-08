#! /bin/bash
#SBATCH -J WGAN
#SBATCH -o ./results/WGAN3.out               
#SBATCH -p compute2             
#SBATCH -A compute2    
#SBATCH --qos=compute2               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python -u -m experiments.models.WGANGP  --datasets Salinas_Corrected