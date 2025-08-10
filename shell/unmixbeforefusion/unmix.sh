#/bin/bash

data=Salinas_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

python experiments/models/Unmixing-before-fusion/Unmixing.py \
    train \
    --batch_size 32 \
    --epochs 100 \
    --dataset_name $data \
    --image_size 128 \
    --save_dir "experiments/models/checkpoints/unmixbeforefusion/" \
    --gpus 1

python experiments/models/Unmixing-before-fusion/Unmixing.py \
    infer \
    --dataset_name $data \
    --image_size 128 \
    --save_dir "experiments/models/checkpoints/unmixbeforefusion/" \
    --gpus 1