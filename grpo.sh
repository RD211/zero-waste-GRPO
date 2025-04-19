#!/bin/bash
#SBATCH --job-name=GRPO
#SBATCH --account=a-large-sc
#SBATCH --partition=debug
#SBATCH --time=00:14:59
#SBATCH --output=/iopsstor/scratch/cscs/%u/GRPO/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/ddinucu/course/full_stack.toml
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

# Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NNODES=$SLURM_NNODES
LOCAL_RANK=$SLURM_LOCALID
ASSIGNMENT_DIR="/iopsstor/scratch/cscs/$USER/GRPO"
CMD_PREFIX="numactl --membind=0-3"
TRAINING_CMD="./start_rl_training.sh --config_file configs/deepspeed/zero3.yaml --config-name llama3.yaml"
WANDB_DIR="/iopsstor/scratch/cscs/$USER/wandb"

bash -c "export LOCAL_RANK=$SLURM_LOCALID ; export HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface ; export WANDB_DIR=/iopsstor/scratch/cscs/$USER/wandb ; cd $ASSIGNMENT_DIR; chmod +x start_rl_training.sh start_vllm_server.sh stop_vllm_server.sh; $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"