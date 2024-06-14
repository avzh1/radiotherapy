#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist loki          	# SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Before I had an sbatch thing above which had --nodelist loki

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

jupyter nbconvert --to script '2_inference.ipynb'

# python3 2_inference.py Anorectum 0
# python3 2_inference.py Anorectum 1
python3 2_inference.py Anorectum 2 # TODO
# python3 2_inference.py Bladder 0
# python3 2_inference.py Bladder 1
# python3 2_inference.py Bladder 2
# python3 2_inference.py CTVn 0
# python3 2_inference.py CTVn 1
# python3 2_inference.py CTVn 2
# python3 2_inference.py CTVp 0
# python3 2_inference.py CTVp 1
# python3 2_inference.py CTVp 2
# python3 2_inference.py Parametrium 0
# python3 2_inference.py Parametrium 1
# python3 2_inference.py Parametrium 2
# python3 2_inference.py Vagina 0
# python3 2_inference.py Vagina 1
# python3 2_inference.py Vagina 2
# python3 2_inference.py Uterus 0
python3 2_inference.py Uterus 1
python3 2_inference.py Uterus 2
