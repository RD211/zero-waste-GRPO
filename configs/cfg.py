from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelvLLMConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    # vLLM settings
    max_length: int = 1024
    max_num_seqs: int = 128

    gpu_memory_utilization: float = 0.8
    number_of_gpus_per_instance: int = 2

    use_v0: bool = True # v1 seems very buggy especially with grace for some reason.
    enforce_eager: bool = True

@dataclass
class Dataset:
    name_or_path: str = "open-r1/Big-Math-RL-Verified-Processed"
    split: str = "train"
    subset: str = "level_3"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    train_datasets: list[Dataset] = field(default_factory=lambda: [Dataset()])
    max_train_examples: int = 1000

@dataclass
class TrainConfig:
    
    # GRPO Specific
    num_problems_per_batch: int = 4 # Number of problems per batch
    group_size: int = 4 # Number of attempts per problem
    beta: float = 0.000004
    loss_type: str = "bnpo" # bnpo, grpo or dr_grpo
    
    normalize_reward: bool = True
    
    
    #TODO: Implement this.
    mu: int = 1
    epsilon: float = 0.2
    ######################
    
    
    gradient_accumulation_steps: int = 1 # We have num_problems_per_batch * group_size / gpus that we can divide.

    learning_rate: float = 1e-5
    lr_warmup_steps: int = 0
    fused_optimizer: bool = False
    max_grad_norm: float = 1.0
    
    max_steps: int = 1000


    gradient_checkpointing: bool = True
    deepspeed_config_path: Optional[str] = 'configs/deepspeed/zero2.yaml'
    
    
    ref_model_inference_batch_size: int = 4
    use_liger_loss: bool = True
    
@dataclass
class LoggingConfig:
    wandb: bool = True
    
    wandb_project: str = "train-grpo"
    wandb_run_name: str = "Qwen-0.5b"
    wandb_entity: str = "rd211"
    
    run_group: str = "7b"
    
    wandb_tags: list[str] = field(default_factory=list)


@dataclass
class CheckpointConfig:
    
    save_steps: int = 100000
    save_total_limit: int = 3
    save_dir: str = "/iopsstor/scratch/cscs/ddinucu/GRPO/checkpoints/qwen-0.5b"
    save_seconds_before_timelimit: int = 60 * 5
    
@dataclass
class RLModelTrainingConfig:
    
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    model: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)
    
    seed: int = 42
    server_port: int = 8005
