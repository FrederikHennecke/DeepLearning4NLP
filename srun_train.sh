#!/bin/bash
# Allocate the resources
# srun -p grete:shared --pty -n 1 -C inet -c 32 -G A100:1 bash
# srun --pty -p grete:interactive -C inet -G 2g.10gb:2 /bin/bash

# # Load anaconda
module load anaconda3
source activate dnlp # Or whatever you called your environment.

# Printing out some info.
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# # For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# # Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Set the parameters
# epochs=1
# batch_size=256
# lr=10
# dropout=0.5

# Run the multitask script:
# python -u multitask_classifier.py --task qqp --option finetune --use_gpu --local_files_only --epochs 1

# Run the bart_detection script:
python -u bart_detection.py --use_gpu --lr 1e-5 --batch_size 64 --epochs 2
