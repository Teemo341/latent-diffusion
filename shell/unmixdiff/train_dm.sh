#/bin/bash

data=Salinas_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

python experiments/models/UnmixingDM/Diffusion.py -p train -c experiments/models/UnmixingDM/config/$data.json
# python experiments/models/UnmixingDM/Diffusion.py val -c experiments/models/UnmixingDM/config/$data.json