#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 05:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:interactive              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task 4            # number of CPU cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=mohamed.aly@stud.uni-goettingen.de  # TODO: change this to your mailaddress!
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
source activate dnlp # Or whatever you called your environment.

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

epochs=1
batch_size=512
lr=10
dropout=0.5
# local_files_only='./models/bert-base-uncased.pt'

# Run the script:
python -u multitask_classifier.py --task sst --epochs $epochs --option pretrain --batch_size $batch_size --lr $lr --hidden_dropout_prob $dropout
# rm slurm*
