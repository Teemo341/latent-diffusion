#/bin/bash

data=PaviaU
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

# python experiments/models/UnmixingDM/Unmixing.py \
#     train \
#     --batch_size 256 \
#     --epochs 10 \
#     --dataset_name $data \
#     --size 32 \
#     --save_dir "experiments/models/checkpoints/UnmixDiff/" \
#     --gpus 2

python experiments/models/UnmixingDM/Unmixing.py \
    infer \
    --dataset_name $data \
    --size 32 \
    --save_dir "experiments/models/checkpoints/UnmixDiff/" \
    --gpus 3