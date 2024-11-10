#! /bin/bash
#SBATCH -J MSTPlus
#SBATCH -o ./results/MSTPlus.out               
#SBATCH -p compute1             
#SBATCH -A compute1    
#SBATCH --qos=compute1               
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1


python -u -m experiments.models.MSTPlus  --datasets Indian_Pines_Corrected