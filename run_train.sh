#!/bin/bash

module load anaconda3
source activate dl-gpu # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

epochs=2
batch_size=128
lr=1e-1
dropout=0.1
sst_filename="./data/sst-sentiment-train.csv"
quora_filename="./data/quora-paraphrase-train.csv"
sts_filename="./data/sts-similar-train.csv"
etpc_filename="./data/etpc-paraphrase-train.csv"

# Run the script:
python -u multitask_classifier.py --task sst --epochs $epochs --sst_train $sst_filename --quora_train $quora_filename --sts_train $sts_filename --etpc_train $etpc_filename \
--option pretrain --use_gpu --batch_size $batch_size --lr $lr --local_files_only --hidden_dropout_prob $dropout
# rm slurm*

# Run
