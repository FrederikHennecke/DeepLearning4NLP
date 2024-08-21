#!/bin/bash
#SBATCH --job-name=train-multitask_classifier
#SBATCH -t 15:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:2                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=16            # number cores per task
#SBATCH --mail-type=END,FAIL              # send mail when job begins and ends
#SBATCH --mail-user=mohamed.aly@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

# srun -p grete:shared --pty -n 1 -C inet -c 16 -G A100:1 --interactive /bin/bash
module load anaconda3
source activate dnlp # Or whatever you called your environment.
# conda install -y -c conda-forge spacy cupy spacy-transformers
# pip install spacy-lookups-data
# python -m spacy download en_core_web_sm


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

# Run the script:
python -u train_multitask.py --use_gpu --local_files_only --option finetune --task multitask --hpo --smoketest --additional_inputs --add_layers --profiler --write_summary --combined_models --use_pal --sst --sts --para
# python -u single_classify.py --use_gpu --local_files_only --option finetune --task sst --smoketest
