#!/bin/bash
#SBATCH --mem 2G
#SBATCH -c 1
#SBATCH -p short-complex
#SBATCH --gpus=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=pdssf@cin.ufpe.br

ENV_NAME=$1
echo "Environment name: $ENV_NAME"
module load Python3.10
python -m venv $HOME/doc/$ENV_NAME
source $HOME/doc/$ENV_NAME/bin/activate
which python
pip install -r ./requirements.txt
python3 intrusion_detection_ch.py