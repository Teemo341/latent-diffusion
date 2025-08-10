#/bin/bash

data=Salinas_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

python experiments/models/UnmixingDM/Unmixing.py \
    train \
    --batch_size 64 \
    --epochs 100 \
    --dataset_name $data \
    --image_size 128 \
    --save_dir "experiments/models/checkpoints/UnmixDiff/" \
    --gpus 3

python experiments/models/UnmixingDM/Unmixing.py \
    infer \
    --dataset_name $data \
    --image_size 128 \
    --save_dir "experiments/models/checkpoints/UnmixDiff/" \
    --gpus 3