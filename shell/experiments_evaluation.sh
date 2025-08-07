#! /bin/bash
#SBATCH -J evaluation
#SBATCH -o ./results/experiments_evaluation.out
#SBATCH -p compute1
#SBATCH -A compute1
#SBATCH --qos=compute1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -w node8

metric="IS FID F_p D_b spectral_curve"
datasets="Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected"
algorithms="UnmixDiff"

python -u -m experiments.metric.evaluation \
    --metric $metric \
    --if_make_original_HSI False \
    --algorithms $algorithms \