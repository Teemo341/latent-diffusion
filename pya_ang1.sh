#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u -m experiments.models.MSTPlus  --datasets Indian_Pines_Corrected 2>&1 | tee ./results/MSTPlus.out
python -u -m experiments.models.MPRNet  --datasets Indian_Pines_Corrected 2>&1 | tee ./results/MPRNet.out
python -u -m experiments.models.MSTPlus1  --datasets KSC_Corrected 2>&1 | tee ./results/MSTPlus1.out