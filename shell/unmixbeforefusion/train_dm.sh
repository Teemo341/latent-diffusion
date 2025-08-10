#/bin/bash

data=Indian_Pines_Corrected
# Indian_Pines_Corrected KSC_Corrected Pavia PaviaU Salinas_Corrected

# python experiments/models/Unmixing-before-fusion/Diffusion.py -p train -c experiments/models/Unmixing-before-fusion/config/$data.json
python experiments/models/Unmixing-before-fusion/Diffusion.py -p val -c experiments/models/Unmixing-before-fusion/config/$data.json