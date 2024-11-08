#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

python -u -m experiments.models.MSTPlus2  --datasets Pavia 2>&1 | tee ./results/MSTPlus2.out
python -u -m experiments.models.MSTPlus3  --datasets PaviaU 2>&1 | tee ./results/MSTPlus3.out
python -u -m experiments.models.MSTPlus4  --datasets Salinas_Corrected 2>&1 | tee ./results/MSTPlus4.out