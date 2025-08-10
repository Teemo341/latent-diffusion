#! /bin/bash
#SBATCH -J test_cuda
#SBATCH -o ./results/test_cuda.out               
#SBATCH -p compute2           
#SBATCH -A compute2    
#SBATCH --qos=compute2                                 
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:5

python -u test.py