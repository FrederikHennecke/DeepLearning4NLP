#!/bin/bash
#SBATCH --job-name=train-multitask_classifier
#SBATCH -t 2:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=12G            # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=TODO@stud.uni-goettingen.de   
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

source activate dnlp # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

mkdir -p models

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

tasks=("sst" "sts" "qqp")
for task in "${tasks[@]}" 
do
    echo "task: $task"
#    python -u multitask_classifier.py --use_gpu --local_files_only --option finetune --task "$task" --hidden_dropout_prob 0.1
done

bart_files=("bart_detection.py")
for file in "${bart_files[@]}"
do
    echo "Running file: $file for etpc tasks"
    python -u "$file" --use_gpu --epochs 20 --lr 1e-5 # --etpc_train "data/etpc-paraphrase-train-minibatch.csv"
    echo
done

