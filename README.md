# Zero Waste GRPO

**Zero Waste GRPO** is a memory-efficient and scalable implementation of GRPO, designed for multi-node training on GH200 systems.

## Highlights

- ⚙️ Supports **DeepSpeed ZeRO-1/2/3** with fine-grained CPU offloading
- 🔁 Automatic checkpointing and **job resubmission** before timeout
- 🧠 Reference model offloading with dynamic reloading for loss computation
- 🚀 Built-in support for **vLLM**-based sampling and inference
- 🧪 Experimental integration with **Transformer Engine FP8 inference**
- 📈 Seamless **WandB tracking** across restarts

## Structure
```
GRPO/
├── configs/ # DeepSpeed + model config files
├── src/
│ ├── grpo/ # Core training logic
│ ├── misc/ # FP8 utilities
│ ├── utils/ # Shared memory, tracker, SLURM utils
│ └── vllm/ # vLLM client, server, inference wrapper
├── rewards.py # Task-specific reward function
├── train_rl.py # Main RL training driver script
├── vllm_server.py # Hackable FastAPI Server
├── Dockerfile # Includes everything needed to run the code
```

## Setup

Before running, create a `.env` file in the root directory with:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

In order to allow resubmission you should also run this once:
```
mkdir -p $HOME/.slurm_env
env | sort > $HOME/.slurm_env/login_env.txt
```

## Usage

On Clariden:
```
sbatch grpo.sh
```

Otherwise:
```
./start_rl_training.sh --config-name qwen14b.yaml \
    --config_file configs/deepspeed/zero3.yaml
```
