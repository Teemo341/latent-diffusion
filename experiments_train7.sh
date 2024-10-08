#! /bin/bash
#SBATCH -J MSTPlus
#SBATCH -o ./results/MSTPlus2.out               
#SBATCH -p compute2             
#SBATCH -A compute2    
#SBATCH --qos=compute2               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1


python -u -m experiments.models.MSTPlus2  --datasets Pavia