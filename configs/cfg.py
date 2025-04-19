from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelvLLMConfig:
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    # vLLM settings
    max_length: int = 8192
    max_num_seqs: int = 256

    gpu_memory_utilization: float = 0.5
    number_of_gpus_per_instance: int = 4

    use_v0: bool = True
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
    group_size: int = 8 # Number of attempts per problem
    beta: float = 0.0001
    mu: int = 4
    epsilon: float = 0.2
    gradient_accumulation_steps: int = 4

    learning_rate: float = 5e-7
    lr_warmup_steps: int = 20
    fused_optimizer: bool = False
    max_grad_norm: float = 1.0
    
    epochs: int = 1
    max_steps: int = -1


    gradient_checkpointing: bool = True
    deepspeed_config_path: Optional[str] = 'configs/deepspeed/zero3.yaml'
    
    
    ref_model_inference_batch_size: int = 2
    
@dataclass
class LoggingConfig:
    wandb: bool = False
    
    wandb_project: str = "train-rl"
    wandb_run_name: str = "Llama3-8B"
    wandb_entity: str = "rd211"
    
    run_group: str = "8b"
    
    wandb_tags: list[str] = field(default_factory=list)


@dataclass
class CheckpointConfig:
    
    save_steps: int = 1000
    save_total_limit: int = 3
          
@dataclass
class RLModelTrainingConfig:
    
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    model: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)
    
    seed: int = 42
    server_port: int = 8005