#!/bin/bash
#SBATCH --job-name=LAI-GRPO
#SBATCH --account=a-large-sc
#SBATCH --partition=normal
#SBATCH --time=00:29:59
#SBATCH --output=/iopsstor/scratch/cscs/%u/GRPO/logs/%x-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/users/ddinucu/course/full_stack.toml

echo "START TIME: $(date)"


MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname -I | awk '{print $1}')
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * 4))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NNODES=$SLURM_NNODES
ulimit -S -c 0

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 --export=ALL bash -lc '
    export LOCAL_RANK=$SLURM_LOCALID
    export RANK=$SLURM_PROCID
    export HF_HOME=/iopsstor/scratch/cscs/$USER/huggingface
    export WANDB_DIR=/iopsstor/scratch/cscs/$USER/wandb
    export HF_HUB_OFFLINE=1
    cd /iopsstor/scratch/cscs/$USER/GRPO
    pip install psutil

    ./start_rl_training.sh --config-name qwen7b.yaml \
        --num_processes $WORLD_SIZE \
        --num_machines $SLURM_NNODES \
        --machine_rank $RANK \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --config_file configs/deepspeed/zero3.yaml
'

echo "END TIME: $(date)"
exit 0