#!/bin/bash
#SBATCH --job-name=GRPO
#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:14:59
#SBATCH --output=/iopsstor/scratch/cscs/%u/GRPO/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/ddinucu/course/full_stack.toml

echo "START TIME: $(date)"

export ASSIGNMENT_DIR=/iopsstor/scratch/cscs/$USER/GRPO
export CMD_PREFIX="numactl --membind=0-3"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -S -c 0

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -lc '
  ulimit -S -c 0
  pip install nvidia-ml-py3
  export VLLM_PORT=$((20000 + (SLURM_JOB_ID % 30000)))
  export LOCAL_RANK=$SLURM_LOCALID
  export RANK=$SLURM_PROCID
  export HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface
  export WANDB_DIR=/iopsstor/scratch/cscs/$USER/wandb
  cd $ASSIGNMENT_DIR
  chmod +x start_rl_training.sh start_vllm_server.sh stop_vllm_server.sh
  '"$CMD_PREFIX"' ./start_rl_training.sh --config_file configs/deepspeed/zero3.yaml --config-name llama3.yaml
'

echo "END TIME: $(date)"
exit 0