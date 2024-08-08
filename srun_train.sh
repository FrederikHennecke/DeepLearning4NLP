#!/bin/bash
# Allocate the resources
# srun -p grete --pty -n 1 -C inet -c 80 -G A100:4 /bin/bash
# srun --pty -p grete:interactive -C inet -G 2g.10gb:2 /bin/bash

# # Load anaconda
module load anaconda3
source activate dnlp # Or whatever you called your environment.
# python -m spacy download en_core_web_sm
# python -m spacy download en



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

# multitasks=("sst" "sts" "qqp")
multitasks=("sst")
bart_files=("bart_detection.py" "bart_generation.py")

# Run the script for all baseline tasks:
# for task in "${multitasks[@]}"
# do
#     echo "Running task: $task"
#     python -u multitask_classifier.py --task "$task" --option finetune --use_gpu --local_files_only
#     echo
# done

# for file in "${bart_files[@]}"
# do
#     echo "Running file: $file for etpc tasks"
#     python -u "$bart_file" --use_gpu
#     echo
# done


# Run tasks individually:
python -u multitask_classifier.py --task sst --option finetune --use_gpu --local_files_only
# python -u bart_detection.py --use_gpu --lr 1e-5 --batch_size 64 --epochs 2
