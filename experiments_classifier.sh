#! /bin/bash
#SBATCH -J classifier
#SBATCH -o ./results/experiments_classifier.out               
#SBATCH -p compute2
#SBATCH -A compute2
#SBATCH --qos=compute2
#SBATCH -N 1               
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

python -u -m experiments.metric.classifier --dataset Salinas_Corrected --embedding_dim 256 --hidden_dim 64 --layers 16 --epochs 50 --lr 1e-4