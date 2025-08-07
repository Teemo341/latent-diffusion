#/bin/bash

data=KSC_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

python /mnt/nas-tianwen/home/ssy/latent-diffusion/experiments/models/UnmixingDM/Synthesis.py \
    --abundance_path logs/unmixdiff/ddpm/KSC_Corrected_250805_202810/mat_results/5000 \
    --AE_path experiments/models/checkpoints/UnmixDiff/UnmixingAE_${data}_latest.pth \
    --dataset_name $data \
    --result_path experiments/results/UnmixDiff/${data}/ \
    --image_path experiments/results/UnmixDiff/${data}/