#/bin/bash

data=Salinas_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

python experiments/models/Unmixing-before-fusion/Synthesis.py \
    --abundance_path logs/Unmixing-before-fusion/ddpm/Salinas_Corrected_250807_180121/mat_results/5469 \
    --AE_path experiments/models/checkpoints/unmixbeforefusion/UnmixingAE_${data}_latest.pth \
    --dataset_name $data \
    --result_path experiments/results/Unmixbeforefusion/${data}/ \
    --image_path experiments/results/Unmixbeforefusion/${data}/