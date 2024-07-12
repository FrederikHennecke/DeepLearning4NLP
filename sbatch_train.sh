#!/bin/bash
#SBATCH --job-name=baseline-finetune-all-tasks
#SBATCH -t 5:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task

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

multitasks=("sst" "sts" "qqp")
bart_files=("bart_detection.py" "bart_generation.py")

# Run the script for bert multitask baseline:
for task in "${multitasks[@]}"
do
  echo "Running task: $task"
  if [ "$task" == "qqp" ]; then
      python -u multitask_classifier.py --task "$task" --option finetune --use_gpu --local_files_only --epochs 2
      echo
  else
      python -u multitask_classifier.py --task "$task" --option finetune --use_gpu --local_files_only
      echo
  fi
done

# Run the script for bart detection and generation baseline:
for file in "${bart_files[@]}"
do
    echo "Running file: $file for etpc tasks"
    python -u "$file" --use_gpu
    echo
done

# Run the script for individual tasks:
# python -u multitask_classifier.py --task "$task" --option finetune --use_gpu --local_files_only
# python -u bart_detection.py --use_gpu --lr 1e-5 --batch_size 64 --epochs 2
