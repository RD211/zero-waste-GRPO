import os
import gc
import copy
import json, shutil, itertools
from datetime import timedelta
from typing import List, Dict, Callable, Sequence
from tracker import gpu_profiler, gpu_tracker

import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator, InitProcessGroupKwargs
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import wandb
from utils import (
    build_lr_scheduler,
    init_logger,
    gather_incremental_state_dict_to_cpu,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from shared_memory import create_shared_state_dict, get_shareable_version
from vllm_client import generate, shutdown
from configs.cfg import RLModelTrainingConfig
from slurm_utils import get_slurm_time_left, received_term
import torch
from accelerate.utils import FP8RecipeKwargs, InitProcessGroupKwargs
from fp8 import fp8ify_module
import transformer_engine.pytorch as te

_orig_load = torch.load


def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _unsafe_load


class RLTrainer:

    def __init__(
        self,
        cfg: RLModelTrainingConfig,
        train_dataset,
        reward_functions: List[Callable],
    ):
        self.cfg = cfg
        self.train_dataset = train_dataset.batch(cfg.train.num_problems_per_batch)
        self.reward_functions = reward_functions

        fp8_kwargs = None
        if self.cfg.train.fp8_training:
            fp8_kwargs = FP8RecipeKwargs(
                backend='te',
                fp8_format="HYBRID",
                amax_compute_algo="max",
                amax_history_len=1024,
                interval=1,
            )
        self.fp8_recipe = fp8_kwargs

        self.accelerator = Accelerator(
            mixed_precision='fp8' if self.cfg.train.fp8_training else 'bf16',
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(hours=1)), # 1 hour timeout for vllm sampling
            ] + ([fp8_kwargs] if fp8_kwargs else []),
        )
        self.device = self.accelerator.device
        self.num_nodes = int(os.environ.get("NNODES", 1))

        self.logger = init_logger(self.accelerator.process_index)
        if self.cfg.logging.wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.cfg.logging.wandb_project,
                name=self.cfg.logging.wandb_run_name,
                entity=self.cfg.logging.wandb_entity,
                group=self.cfg.logging.run_group,
                tags=self.cfg.logging.wandb_tags,
                config=OmegaConf.to_object(self.cfg),
            )

        train_cfg = self.cfg.train
        model_cfg = self.cfg.model

        self.rank = self.accelerator.process_index
        self.gpus_per_node = self.accelerator.state.num_processes // self.num_nodes
        self.node_id = self.rank // self.gpus_per_node

        ds_plugin = self.accelerator.state.deepspeed_plugin
        ds_cfg = ds_plugin.deepspeed_config

        ds_cfg["train_batch_size"] = (
            train_cfg.num_problems_per_batch * train_cfg.group_size
        )
        ds_cfg["gradient_accumulation_steps"] = train_cfg.gradient_accumulation_steps
        per_proc = ds_cfg["train_batch_size"] // self.accelerator.state.num_processes
        ds_cfg["train_micro_batch_size_per_gpu"] = (
            per_proc // train_cfg.gradient_accumulation_steps
        )

        assert ds_cfg["train_batch_size"] % self.accelerator.state.num_processes == 0
        assert per_proc % train_cfg.gradient_accumulation_steps == 0

        self._current_step = 0

        # ------------------------------------------------------------------
        # Model setup
        # ------------------------------------------------------------------

        if self.accelerator.is_main_process:
            snapshot_download(model_cfg.model_name_or_path)
        self.accelerator.wait_for_everyone()

        # Hot model
        auto_model_class = (
            AutoLigerKernelForCausalLM
            if self.cfg.train.use_liger_loss
            else AutoModelForCausalLM
        )
        self.model = auto_model_class.from_pretrained(
            model_cfg.model_name_or_path,
            use_cache=not train_cfg.gradient_checkpointing,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if self.cfg.train.fp8_training:
            fp8ify_module(
                self.model
            )
        print(f"Model {self.model} loaded")
        self.logger.info(f"Hot model loaded.")

        # Reference model
        self.ref_model = auto_model_class.from_pretrained(
            model_cfg.model_name_or_path,
            use_cache=not train_cfg.gradient_checkpointing,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        ref_ds_cfg = copy.deepcopy(ds_cfg)
        ref_ds_cfg.update(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {"device": "cpu", "pin_memory": True},
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "overlap_comm": False,
                    "contiguous_gradients": False,
                },
                "optimizer": {"type": None},
            }
        )

        self.ref_model = deepspeed.initialize(
            model=self.ref_model,
            config=ref_ds_cfg,
            optimizer=None,
            lr_scheduler=None,
        )[0]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        if train_cfg.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        # ------------------------------------------------------------------
        # Optimizer and scheduler
        # ------------------------------------------------------------------

        self.optimizer = deepspeed.ops.adam.FusedAdam(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            adam_w_mode=True,
        )
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, train_cfg.lr_warmup_steps
        )

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------

        self.loss_fn = LigerFusedLinearGRPOLoss(
            beta=train_cfg.beta,
            use_ref_model=True,
            loss_type=cfg.train.loss_type,
            temperature=model_cfg.temperature,
        )

        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------

        self.train_iterator = iter(self.train_dataset)

    def train(self):

        self.logger.info("Starting training")
        start = self._current_step

        i = 0
        while i < start:
            next(self.train_iterator)
            i += 1

        for step, batch in enumerate(self.train_iterator, start=start):
            if self.cfg.train.max_steps > 0 and step > self.cfg.train.max_steps:
                self.logger.info("Reached max_steps -> stopping")
                break

            prompts = batch["prompt"]
            solutions = batch["solution"]

            self.logger.info("Gathering state dict for vLLM sampling")
            state_dict = gather_incremental_state_dict_to_cpu(
                self.accelerator, self.logger, self.model
            )
            self._offload_optimizer_and_policy()

            g_size = self.cfg.train.group_size
            dup_prompts = [p for p in prompts for _ in range(g_size)]
            dup_solutions = [s for s in solutions for _ in range(g_size)]

            gens = self._sample_with_vllm(state_dict, dup_prompts)
            self.logger.info(f"Node {self.node_id}: sampled {len(gens)} generations")

            self.logger.info("Computing rewards and advantages")

            self.logger.info(
                f"Prompts: {len(prompts)}, with number of duplicates: {len(dup_solutions)}"
            )
            rewards = self._compute_rewards(gens, dup_solutions)
            advantages = self._compute_advantages(rewards, g_size)
            self.logger.info(
                f"gens: {[len(g) for g in gens]}, rewards: {rewards}, advantages: {advantages}"
            )

            self.logger.info("Computing reference logps")
            conversations = [
                [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": g},
                ]
                for p, g in zip(dup_prompts, gens)
            ]
            txt = self.tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            ids = self.tokenizer(txt, add_special_tokens=False)["input_ids"]

            ref_logps = self._compute_reference_logps(ids)
            # We assert the lengths of the ids are bigger by 1 than the ref_logps
            for i, (id_seq, ref_seq) in enumerate(zip(ids, ref_logps)):
                assert len(id_seq) == len(ref_seq) + 1, f"seq {i} has wrong length"

            self.logger.info("Updating policy")
            self._reload_optimizer_and_policy()
            loss, kl, token_len = self._policy_update(ids, ref_logps, advantages)

            self.logger.info(
                f"Step {step}: loss: {loss:.4f}, kl: {kl:.4f}, reward: {torch.mean(torch.tensor(rewards)):.4f}, token_len: {token_len:.4f}"
            )

            if self.cfg.logging.wandb and self.accelerator.is_main_process:

                wandb.log(
                    {
                        "reward": torch.mean(torch.tensor(rewards)),
                        "loss": loss,
                        "kl": kl,
                        "completion_length": token_len,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "step": step,
                    },
                    step=step,
                )

            if step % self.cfg.checkpoint.save_steps == 0 and step > 0:
                self._save_checkpoint()
                self.logger.info(f"Checkpoint saved at step {step}")

            # If we got slurm kill signal, we checkpoint and exit.
            if (
                received_term()
                or get_slurm_time_left()
                < self.cfg.checkpoint.save_seconds_before_timelimit
            ):
                self.logger.info(
                    "Received kill signal or time. Saving checkpoint and exiting."
                )
                self._save_checkpoint()
                break

            self._current_step += 1

        if self.accelerator.is_local_main_process:
            try:
                shutdown()
            except Exception as e:
                pass

        if not (
            received_term()
            or get_slurm_time_left() < self.cfg.checkpoint.save_seconds_before_timelimit
        ):
            self.logger.info("Training complete!")

        wandb.finish()

        return not (
            received_term()
            or get_slurm_time_left() < self.cfg.checkpoint.save_seconds_before_timelimit
        )

    @gpu_profiler("vllm_sampling", no_memory_measurement=True)
    def _sample_with_vllm(
        self,
        state_dict: Dict[str, torch.Tensor],
        prompts: Sequence[str],
    ) -> List[str]:
        per_node = len(prompts) // self.num_nodes

        # Slice prompts belonging to this node
        local_prompts = prompts[self.node_id * per_node : (self.node_id + 1) * per_node]
        self.logger.info(
            f"Node {self.node_id}: sampling {len(local_prompts)} generations"
        )

        gens: List[str] = [None] * per_node
        if self.accelerator.is_local_main_process:
            meta = create_shared_state_dict(state_dict)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Meta created and delete old state_dict")
            shared_meta = get_shareable_version(meta)
            self.logger.info("Meta shared")
            gens = generate(local_prompts, shared_meta)
            gens = [(self.node_id, i, g) for i, g in enumerate(gens)]

        self.accelerator.wait_for_everyone()
        gens_all = self.accelerator.gather_for_metrics(gens)
        gens_all = [g for g in gens_all if g is not None]
        gens_all.sort(key=lambda x: (x[0], x[1]))
        return [g[2] for g in gens_all]

    @gpu_profiler("compute_rewards")
    def _compute_rewards(
        self, generations: List[str], answers: List[str]
    ) -> List[float]:
        rewards_per_gen = [0.0] * len(generations)
        for fn in self.reward_functions:
            r = [fn(completion=g, answer=a) for g, a in zip(generations, answers)]
            for i, val in enumerate(r):
                rewards_per_gen[i] += val

        # average over reward functions
        n_fn = len(self.reward_functions)
        return [val / n_fn for val in rewards_per_gen]
    
    def _compute_advantages(self, rewards: List[float], group_size: int) -> List[float]:
        grouped = [
            rewards[i : i + group_size] for i in range(0, len(rewards), group_size)
        ]
        advantages = []
        for g in grouped:
            mean = sum(g) / group_size
            std = (sum((x - mean) ** 2 for x in g) / group_size) ** 0.5
            advantages.extend(
                [
                    (r - mean)
                    / ((std + 1e-8) if self.cfg.train.normalize_reward else 1)
                    for r in g
                ]
            )
        return advantages

    @gpu_profiler("compute_reference_logps")
    def _compute_reference_logps(self, ids: List[List[int]]) -> List[List[float]]:

        per_rank = len(ids) // self.accelerator.state.num_processes
        start, end = self.rank * per_rank, (self.rank + 1) * per_rank
        ids_local = ids[start:end]

        bs = self.cfg.train.ref_model_inference_batch_size
        ref_logps_local: List[List[float]] = []

        for i in range(0, len(ids_local), bs):
            batch_ids = ids_local[i : i + bs]
            lens = [len(seq) for seq in batch_ids]
            padded = self.tokenizer.pad(
                {"input_ids": batch_ids},
                padding="longest",
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self.device)
            _, L = input_ids.shape
            seq_range = torch.arange(L, device=self.device).unsqueeze(0)
            attention_mask = (
                seq_range < torch.tensor(lens, device=self.device).unsqueeze(1)
            ).long()

            with torch.no_grad():
                logits = (
                    self.ref_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                    / self.cfg.model.temperature
                )

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            shift_lp = log_probs[:, :-1, :]
            shift_lbl = input_ids[:, 1:]
            tok_logps = shift_lp.gather(-1, shift_lbl.unsqueeze(-1)).squeeze(-1)
            tok_logps = tok_logps * (attention_mask[:, 1:] == 1)

            for row, ln in zip(tok_logps, lens):
                ref_logps_local.append(row[: ln - 1].cpu().tolist())

        gathered = self.accelerator.gather_for_metrics(
            list(
                enumerate(ref_logps_local, start)
            )  # TODO: Unsure if this is needed at all. They may be aready ordered.
        )
        gathered.sort(key=lambda x: x[0])
        return [g[1] for g in gathered]

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    @gpu_profiler("torch_loss")
    def _compute_torch_loss(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_tensor: torch.Tensor,
        adv_tensor: torch.Tensor,
        selected_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[float]]:

        B, L = input_ids.shape
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=L,
            ).logits

            logits = logits[:, :-1, :]
            logits = logits / self.cfg.model.temperature

            logps = torch.nn.functional.log_softmax(logits, dim=-1)
            per_tok_logps = logps.gather(-1, selected_token_ids.unsqueeze(-1)).squeeze(-1)

            per_tok_kl = (
                torch.exp(ref_tensor - per_tok_logps) - (ref_tensor - per_tok_logps) - 1.0
            )

            old_per_tok_logps = per_tok_logps.detach()
            ratio = torch.exp(per_tok_logps - old_per_tok_logps)
            eps = self.cfg.train.epsilon
            ratio_clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

            adv_expanded = adv_tensor.unsqueeze(1).expand_as(per_tok_logps)

            loss_1 = ratio * adv_expanded
            loss_2 = ratio_clipped * adv_expanded
            per_tok_loss = -torch.min(loss_1, loss_2)

            beta = self.cfg.train.beta
            if beta != 0.0:
                per_tok_loss = per_tok_loss + beta * per_tok_kl

            token_mask = attention_mask[:, 1:].to(per_tok_loss.dtype)

            loss = (per_tok_loss * token_mask).sum() / token_mask.sum()

            mean_kl = ((per_tok_kl * token_mask).sum() / token_mask.sum()).item()

        return loss, (mean_kl,)

    def _policy_update(
        self,
        ids: List[List[int]],
        ref_logps: List[List[float]],
        advantages: List[float],
    ) -> tuple[float, float, float]:

        per_rank = len(ids) // self.accelerator.state.num_processes
        start, end = self.rank * per_rank, (self.rank + 1) * per_rank
        ids_local = ids[start:end]
        ref_logps_local = ref_logps[start:end]
        adv_local = advantages[start:end]

        # all must have the same length
        assert len(ids_local) == len(ref_logps_local) == len(adv_local)

        micro_bs = (
            self.cfg.train.num_problems_per_batch
            * self.cfg.train.group_size
            // self.accelerator.state.num_processes
            // self.cfg.train.gradient_accumulation_steps
        )
        total_loss, total_kl, total_tokens = 0.0, 0.0, 0

        unwrapped = self.accelerator.unwrap_model(self.model)

        for mb_start in range(0, len(ids_local), micro_bs):
            mb_end = min(mb_start + micro_bs, len(ids_local))

            ids_mb = ids_local[mb_start:mb_end]
            ref_mb = ref_logps_local[mb_start:mb_end]
            adv_mb = adv_local[mb_start:mb_end]

            lengths = [len(seq) for seq in ids_mb]
            padded = self.tokenizer.pad(
                {"input_ids": ids_mb}, padding="longest", return_tensors="pt"
            )
            input_ids = padded["input_ids"].to(self.device)
            B, L = input_ids.shape

            seq_range = torch.arange(L, device=self.device).unsqueeze(0)
            attn_mask = (
                seq_range < torch.tensor(lengths, device=self.device).unsqueeze(1)
            ).long()

            tgt_ids = input_ids[:, 1:]

            ref_tensor = torch.zeros((B, L - 1), device=self.device)
            for i, row in enumerate(ref_mb):
                ref_tensor[i, : len(row)] = torch.tensor(row, device=self.device)

            adv_tensor = torch.tensor(adv_mb, device=self.device, dtype=torch.float32)
            loss, metrics = None, None
            if self.cfg.train.use_liger_loss:

                gp = deepspeed.zero.GatheredParameters(
                    [unwrapped.lm_head.weight, unwrapped.lm_head.bias],
                    modifier_rank=None,
                )
                gp.__enter__()

                with gpu_tracker(
                    "liger_loss",
                    self._current_step,
                    log_to_wandb=self.cfg.logging.wandb,
                    accel=self.accelerator,
                ):
                    hidden = unwrapped.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                    ).last_hidden_state[:, :-1, :]

                    tgt_ids = input_ids[:, 1:]
                    msk = attn_mask[:, 1:]

                    loss, metrics = self.loss_fn(
                        _input=hidden,
                        lin_weight=unwrapped.lm_head.weight,
                        selected_token_ids=tgt_ids,
                        attention_mask=msk,
                        bias=unwrapped.lm_head.bias,
                        advantages=adv_tensor,
                        ref_per_token_logps=ref_tensor,
                    )

                self.accelerator.backward(loss)
                gp.__exit__(None, None, None)

            else:
                loss, metrics = self._compute_torch_loss(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    ref_tensor=ref_tensor,
                    adv_tensor=adv_tensor,
                    selected_token_ids=tgt_ids,
                )
                self.accelerator.backward(loss)

            total_loss += loss.item() * B
            total_kl += metrics[0] * B
            total_tokens += sum(lengths)

            gc.collect()
            torch.cuda.empty_cache()

        self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.max_grad_norm
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        loss_tensor = torch.tensor([total_loss], device=self.device)
        kl_tensor = torch.tensor([total_kl], device=self.device)
        tok_tensor = torch.tensor([total_tokens], device=self.device)

        loss_sum = self.accelerator.reduce(loss_tensor, reduction="sum").item()
        kl_sum = self.accelerator.reduce(kl_tensor, reduction="sum").item()
        tok_sum = self.accelerator.reduce(tok_tensor, reduction="sum").item()

        global_B = len(ids)
        return (
            loss_sum / global_B,
            kl_sum / global_B,
            tok_sum / global_B,
        )

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        step = int(checkpoint_path.split("_")[-1]) + 1
        self._current_step = step

        self.accelerator.load_state(checkpoint_path)
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def _save_checkpoint(self) -> None:
        out_dir = self.cfg.checkpoint.save_dir
        path = os.path.join(out_dir, f"step_{self._current_step}")

        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(path)

        if self.accelerator.is_main_process:

            ckpts = sorted(
                [p for p in os.listdir(out_dir) if p.startswith("step_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            while len(ckpts) > self.cfg.checkpoint.save_total_limit:
                shutil.rmtree(os.path.join(out_dir, ckpts.pop(0)), ignore_errors=True)

        self.logger.info(f"Checkpoint saved to {path}")

    # ------------------------------------------------------------------
    # Offload stuff to CPU.
    # ------------------------------------------------------------------
    @gpu_profiler("offload_optimizer_and_policy")
    def _offload_optimizer_and_policy(self):
        from deepspeed.runtime.zero.offload_config import (
            OffloadDeviceEnum,
            OffloadStateTypeEnum,
        )

        # If we are in deepspeed 3
        if not hasattr(self.model.optimizer, "offload_states"):
            return

        self.logger.info("Offloading optimizer and policy to CPU")
        self.model.optimizer.offload_states(
            include=[
                OffloadStateTypeEnum.optim_states,
                OffloadStateTypeEnum.contiguous_grad_buffer,
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.lp_params,
                OffloadStateTypeEnum.lp_grads,
            ],
            device=OffloadDeviceEnum.cpu,
            pin_memory=True,
            non_blocking=True,
        )
        self.model.empty_partition_cache()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        self.logger.info("Optimizer and policy offloaded to CPU")

    @gpu_profiler("reload_optimizer_and_policy")
    def _reload_optimizer_and_policy(self):

        if not hasattr(self.model.optimizer, "offload_states"):
            return
        
        self.logger.info("Reloading optimizer and policy from CPU")
        self.model.optimizer.reload_states(non_blocking=True)
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        self.logger.info("Optimizer and policy reloaded from CPU")
