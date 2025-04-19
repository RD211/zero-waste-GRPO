import os
import time

import wandb
import hydra
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from huggingface_hub import snapshot_download
from accelerate import Accelerator
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from loss_utils import get_batch_logps, unpad_logps
from utils import build_lr_scheduler, init_logger, prepare_deepspeed, gather_incremental_state_dict_to_cpu
from configs.cfg import RLModelTrainingConfig
from utils import load_datasets
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from vllm import SamplingParams
from vllm_client import generate
from rewards import dumb_reward_func
from shared_memory import create_shared_state_dict, get_shareable_version
from dotenv import load_dotenv
load_dotenv()

kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=10))]
accelerator = Accelerator(kwargs_handlers=kwargs)
init_logger(accelerator.process_index)
cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

@hydra.main(config_path="configs", version_base=None)
def train(cfg: RLModelTrainingConfig):
  

  #--------------------------------------------------------
  # Setup
  #--------------------------------------------------------
  
  logger = init_logger(accelerator.process_index)
  number_of_nodes = os.environ.get("NNODES", 1)
  device = accelerator.device

  # We merge the config with the defaults
  default_config = OmegaConf.structured(RLModelTrainingConfig)

  # Merge loaded config with defaults
  cfg: RLModelTrainingConfig = OmegaConf.merge(default_config, cfg)  
  
  # Configs
  model_config = cfg.model
  train_config = cfg.train
  logging_config = cfg.logging
  data_config = cfg.dataset
  checkpoint_config = cfg.checkpoint

  set_seed(cfg.seed)

  deepspeed_plugin = accelerator.state.deepspeed_plugin
  config_kwargs = deepspeed_plugin.deepspeed_config
  config_kwargs["train_batch_size"] = train_config.num_problems_per_batch * train_config.group_size
  config_kwargs["gradient_accumulation_steps"] = train_config.gradient_accumulation_steps
  config_kwargs["train_micro_batch_size_per_gpu"] = train_config.num_problems_per_batch * train_config.group_size // accelerator.state.num_processes // train_config.gradient_accumulation_steps
  
  # We do some config checks.
  assert config_kwargs['train_batch_size'] % accelerator.state.num_processes == 0, f"Train batch size {config_kwargs['train_batch_size']} is not divisible by number of processes {accelerator.state.num_processes}"
  assert (config_kwargs['train_batch_size'] // accelerator.state.num_processes) % train_config.gradient_accumulation_steps == 0, f"Cannot divide global batch size on processes and gradient accumulation steps. {config_kwargs['train_batch_size'] // accelerator.state.num_processes} % {train_config.gradient_accumulation_steps} != 0"

  
  if logging_config.wandb and accelerator.is_main_process:
      wandb.init(
          project=logging_config.wandb_project,
          name=logging_config.wandb_run_name,
          entity=logging_config.wandb_entity,
          group=logging_config.run_group,
          tags=logging_config.wandb_tags,
          config=OmegaConf.to_object(cfg),
      )

  logger.info('Downloading model')
  if accelerator.is_main_process:
    snapshot_download(
        model_config.model_name_or_path,
    )
  accelerator.wait_for_everyone()
    
    
  #--------------------------------------------------------
  # Load ref & hot model
  #--------------------------------------------------------
  
  logger.info("Loading ref & hot model...")
  model = AutoLigerKernelForCausalLM.from_pretrained(
    model_config.model_name_or_path,
    use_cache=not train_config.gradient_checkpointing,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path,
    trust_remote_code=True,
  )
  
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

  if train_config.gradient_checkpointing:
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
  
  ref_model = AutoLigerKernelForCausalLM.from_pretrained(
    model_config.model_name_or_path,
    use_cache=not train_config.gradient_checkpointing,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
  )
  
  
  ref_model = prepare_deepspeed(ref_model, accelerator)
  
  accelerator.wait_for_everyone()
  
  # TODO: Offload ref model to CPU
  
  #--------------------------------------------------------
  # Load datasets
  #--------------------------------------------------------
  
  logger.info('Getting datasets')
  train_dataset = load_datasets(data_config, cfg.seed)
  
  # We apply the template to the dataset
  def get_apply_conv_template_fn(tokenizer):
    def apply_conv_template(example):
      prompt = example["prompt"]
      solution = example["solution"]
      
      prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
      ], tokenize=False, add_generation_prompt=True)
      
      return {
        "prompt": prompt,
        "solution": solution,
      }
    return apply_conv_template
  
  train_dataset = train_dataset.map(get_apply_conv_template_fn(tokenizer))
    
  train_dataset = train_dataset.batch(train_config.num_problems_per_batch)
  train_iterator = iter(train_dataset)
  
  accelerator.wait_for_everyone()

  #--------------------------------------------------------
  # Optimizer and scheduler
  #--------------------------------------------------------
  
  logger.info("Creating optimizer and scheduler")
  optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, fused=train_config.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, train_config.lr_warmup_steps)
  
  model, optimizer = accelerator.prepare(
      model,
      optimizer,
  )
  
  loss_fn = LigerFusedLinearGRPOLoss(
    beta=train_config.beta,
    use_ref_model=True,
    loss_type='grpo',
    temperature=model_config.temperature,
  )
  accelerator.wait_for_everyone()
    
  #--------------------------------------------------------
  # Reward functions
  #--------------------------------------------------------
  
  logger.info("Creating reward functions")
  reward_functions = [
    dumb_reward_func,
  ]
  
  #--------------------------------------------------------
  # Training
  #--------------------------------------------------------
  
  logger.info("Starting training!")
  
  for batch_idx, batch in enumerate(train_iterator):
    
    # TODO: Offload model and optimizer to CPU
    
    # Stop training if we reach the max steps
    if train_config.max_steps != -1 and batch_idx >= train_config.max_steps:
      logger.info("Max steps reached, stopping training")
      break
    
    
    batch_prompts = batch["prompt"] # Identical for all ranks
    batch_solutions = batch["solution"]
    
    #--------------------------------------------------------    
    # vLLM sampling
    #--------------------------------------------------------
    
    # We need to first gather the state dict locally.
    logger.info("Gathering state dict")
    state_dict = gather_incremental_state_dict_to_cpu(
      accelerator,
      logger=logger,
      model=model,
    )
    
    number_of_prompts_per_node = len(batch_prompts) * train_config.group_size // number_of_nodes
    duplicated_batch_prompts = [prompt for prompt in batch_prompts for _ in range(train_config.group_size)]
    duplicated_batch_solutions = [solution for solution in batch_solutions for _ in range(train_config.group_size)]
    number_of_gpus_per_node = accelerator.state.num_processes // number_of_nodes
    generations_str = [None]*number_of_prompts_per_node

    # We only sample on the local main processes.
    if accelerator.is_local_main_process:
      # We need to split the batch prompts
      node_id = accelerator.process_index // number_of_gpus_per_node
      start_idx = node_id * number_of_prompts_per_node
      end_idx = start_idx + number_of_prompts_per_node
      local_batch_prompts = duplicated_batch_prompts[start_idx:end_idx]

      logger.info(f"Node {node_id} sampling {len(local_batch_prompts)} prompts")

      # We create a shareable/serializable sampling params object
      meta_info = create_shared_state_dict(state_dict)
      meta_info_shared = get_shareable_version(meta_info)
      del state_dict
      torch.cuda.empty_cache()

      generations_str = generate(
        local_batch_prompts,
        meta_info_shared,
      )
      generations_str = [(node_id, g) for g in generations_str]
      logger.info(f"Node {node_id} finished sampling, got {len(generations_str)} generations")
    else:
      del state_dict

    accelerator.wait_for_everyone()

    # We need to synchronize the generations and put them in the right order.
    all_generations_str = accelerator.gather_for_metrics(generations_str)
    # Filter out the None values
    all_generations_str = [g for g in all_generations_str if g is not None]
    # We sort by the first element of the tuple
    all_generations_str.sort(key=lambda x: x[0])
    # We only keep the second element of the tuple
    all_generations_str = [g[1] for g in all_generations_str]

    logger.info(f"Gathered this many generations: {len(all_generations_str)}")
    
    #---------------------------------------------------------
    # Compute rewards
    #---------------------------------------------------------
    logger.info("Computing rewards")
    all_rewards = []
    reward_per_function = {}
    for reward_function in reward_functions:
      reward_per_function[reward_function.__name__] = []
      for generation, solution in zip(all_generations_str, duplicated_batch_solutions):
        reward = reward_function(
          completion=generation,
          answer=solution,
        )
        reward_per_function[reward_function.__name__].append(reward)
        
    # We compute all_rewards which is average over the reward functions for each generation.
    for i in range(len(all_generations_str)):
      all_rewards.append(sum([reward_per_function[r][i] for r in reward_per_function]) / len(reward_per_function))
      
    logger.info(f"Got {len(all_rewards)} rewards with these values: {all_rewards}")
    assert len(all_rewards) == len(all_generations_str)
    
    #--------------------------------------------------------
    # Compute advantages
    #--------------------------------------------------------
    
    # Each generation is part of a problem.
    rewards_grouped_by_problem = []
    for i in range(0, len(all_rewards), train_config.group_size):
      rewards_grouped_by_problem.append(all_rewards[i:i+train_config.group_size])
    # We compute the mean and std of the rewards for each problem.
    rewards_grouped_by_problem = [
      {
        "mean": sum(r) / len(r),
        "std": (sum([(x - sum(r) / len(r))**2 for x in r]) / len(r))**0.5,
      }
      for r in rewards_grouped_by_problem
    ]
    # We compute the advantages.
    all_advantages = []
    for i in range(len(all_rewards)):
      rewards = rewards_grouped_by_problem[i // train_config.group_size]
      mean = rewards["mean"]
      std = rewards["std"]
      advantage = (all_rewards[i] - mean) / (std + 1e-8) # We can ommit the std if we want Dr.GRPO
      all_advantages.append(advantage)
    
    assert len(all_advantages) == len(all_generations_str)
    
    #--------------------------------------------------------
    # Compute ref logps
    #--------------------------------------------------------

    all_conversations = [
      [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generation},
      ]
      for prompt, generation in zip(duplicated_batch_prompts, all_generations_str)
    ]

    all_conversations_str = tokenizer.apply_chat_template(
      all_conversations,
      tokenize=False,
      add_generation_prompt=False,
    )

    completion_ids = tokenizer(
      all_conversations_str, add_special_tokens=False # If we add special tokens we get double bos tokens :)
    )['input_ids']
      
    logger.info("Tokenized conversations")
    logger.info(f"Got {len(completion_ids)} completions with sizes {[len(c) for c in completion_ids]}")
    
    #TODO: Load ref model and optimizer from CPU. Maybe load optimizer async while computing ref logps.
    
    # We now compute logps for the ref_model.
    ref_token_probs = []

    number_of_completions_per_process = len(completion_ids) // accelerator.state.num_processes
    
    # We get our slice.
    start_idx = accelerator.process_index * number_of_completions_per_process
    end_idx = start_idx + number_of_completions_per_process
    local_completion_ids = completion_ids[start_idx:end_idx]
    
    ref_logits_batch_size = train_config.ref_model_inference_batch_size
    # We split the completion_ids by number of processes.
    for i in range(0, len(local_completion_ids), ref_logits_batch_size):
      completion_ids_batch = local_completion_ids[i:i+ref_logits_batch_size]

      lengths = [len(seq) for seq in completion_ids_batch]

      padded = tokenizer.pad(
          {"input_ids": completion_ids_batch},
          padding="longest",
          return_tensors="pt",
      )
      input_ids = padded["input_ids"]
      B, L = input_ids.shape

      lengths_tensor = torch.tensor(lengths, device=input_ids.device)  # [B]
      arange = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)  # [1,L]→[B,L]
      attention_mask = (arange < lengths_tensor.unsqueeze(1)).long()  # [B,L]

      with torch.no_grad():
        logps_padded = get_batch_logps(
            ref_model,
            input_ids=input_ids.to(accelerator.device),
            attention_mask=attention_mask.to(accelerator.device),
            temperature=model_config.temperature,
        )

      logps = unpad_logps(logps_padded, attention_mask.cpu())

        
      # We add indices to the logps in pairs.
      ref_token_probs.extend(
        list(zip(
          list(range(start_idx + i, start_idx + i + len(completion_ids_batch))),
          logps,
        ))
      )
      
      assert len(logps) == len(completion_ids_batch), f"Got {len(logps)} ref logps but expected {len(completion_ids_batch)}"
        
    accelerator.wait_for_everyone()
    
    # We now sync the ref_token_probs.
    all_ref_token_probs = accelerator.gather_for_metrics(ref_token_probs)
    # Sort by the first element of the tuple
    all_ref_token_probs.sort(key=lambda x: x[0]) # No clue if I get a guarantee that this is ordered by rank or not.
    # We only keep the second element of the tuple
    all_ref_token_probs = [g[1] for g in all_ref_token_probs]
    
    
    #---------------------------------------------------------
    # Compute losses
    #---------------------------------------------------------
    
    # We have:
    # - all_ref_token_probs: list of lists of logps for each generation
    # - all_advantages: list of advantages for each generation
    # - completion_ids: list of lists of token ids for each generation
    
    # We need to get our slice.
    start_idx = accelerator.process_index * number_of_completions_per_process
    end_idx = start_idx + number_of_completions_per_process
    local_ref_token_probs = all_ref_token_probs[start_idx:end_idx]
    local_advantages = all_advantages[start_idx:end_idx]
    local_completion_ids = completion_ids[start_idx:end_idx]
    
    local_batch_size = config_kwargs['train_micro_batch_size_per_gpu']
    
    for i in range(0, len(local_ref_token_probs), local_batch_size):
      ref_token_probs_batch = local_ref_token_probs[i:i+local_batch_size]
      advantages_batch = local_advantages[i:i+local_batch_size]
      completion_ids_batch = local_completion_ids[i:i+local_batch_size]
      
      lengths = [len(c) for c in completion_ids_batch]  

      padded = tokenizer.pad(
          {"input_ids": completion_ids_batch},
          padding="longest",
          return_tensors="pt",
      )
      input_ids = padded["input_ids"].to(device)
      B, L = input_ids.shape
      
      # We pad each ref_token_probs to L
      ref_token_probs_batch = [
        logps + [0] * (L - 1 - len(logps))
        for logps in ref_token_probs_batch
      ]

      ref_token_probs_batch = torch.tensor(ref_token_probs_batch, device=device)

      lengths_tensor = torch.tensor(lengths, device=device)    # [B]
      seq_range     = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
      attention_mask = (seq_range < lengths_tensor.unsqueeze(1)).long().to(device)        # [B, L]


      logger.info("Unwrapping model")
      unwrapped_model = accelerator.unwrap_model(model)
      logger.info("Unwrapped model")
      # We need to compute the last_hidden_states
      last_hidden_states = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
      logger.info("Computed last hidden states")
      last_hidden_states = last_hidden_states[:, :-1, :]
      input_ids = input_ids[:, 1:]
      attention_mask = attention_mask[:, 1:]

      with deepspeed.zero.GatheredParameters(
              [unwrapped_model.lm_head.weight, unwrapped_model.lm_head.bias],
              modifier_rank=None):  
        loss, metrics = loss_fn(
            _input=last_hidden_states,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            bias=unwrapped_model.lm_head.bias,
            advantages=torch.tensor(advantages_batch, device=device),
            ref_per_token_logps=ref_token_probs_batch,
        )

        logger.info(f"Loss: {loss.item()}, metrics: {metrics}")
        accelerator.backward(loss)
    
    # We step
    accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    # TODO: Offload ref model to CPU

    # TODO: Load model from CPU.
    # TODO: Compute loss.
    # TODO: Offload model to CPU.
  
  # # Utils
  # num_flop_per_token = get_num_flop_per_token(
  #     get_num_params(model, exclude_embedding=True),
  #     model_config,
  # )

  # ntokens_since_last_log = 0
  # ntraining_tokens_since_last_log = 0
  # time_last_log = time.perf_counter()

  # logger.info("Starting training!")
  # train_step = 0
  # while train_step < args.training_steps:
  #   train_step += 1

  #   # Profiling
  #   if args.profile and args.profile_step_start == train_step:
  #     torch.cuda.cudart().cudaProfilerStart()
  #     torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

  #   input_ids, labels = next(train_dl_iterator)
  #   ntokens_since_last_log += args.batch_size * args.sequence_length
  #   num_items_in_batch = labels.ne(-100).sum()
  #   ntraining_tokens_since_last_log += num_items_in_batch
  #   input_ids = input_ids.to(device)
  #   labels = labels.to(device)

  #   optimizer.zero_grad()

  #   logits = model(input_ids)
  #   loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
  #   loss = loss / num_items_in_batch
  #   del logits
  #   loss.backward()

  #   # Clip gradients
  #   clip_grad_norm_(model.parameters(), args.grad_max_norm)

  #   optimizer.step()
  #   lr_scheduler.step()

  #   # Logging
  #   if (train_step == 1 or train_step % args.logging_frequency == 0):
  #     time_delta = time.perf_counter() - time_last_log
  #     # tokens per second per device, abbreviated as tps
  #     tps = ntokens_since_last_log / time_delta 
  #     mfu = 100 * num_flop_per_token * tps / 989e12
  #     tflops = num_flop_per_token * tps / 1e12
  #     training_tps = ntraining_tokens_since_last_log / time_delta

  #     logger.info(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
  #     ntokens_since_last_log = 0
  #     ntraining_tokens_since_last_log = 0
  #     time_last_log = time.perf_counter()
    
  #   # Profiling
  #   if args.profile and args.profile_step_end == train_step:
  #     torch.cuda.cudart().cudaProfilerStop()


  logger.info("Training completed")

if __name__ == "__main__":
  train()
