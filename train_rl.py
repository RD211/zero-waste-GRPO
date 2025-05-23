import os
import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer, set_seed

from configs.cfg import RLModelTrainingConfig
from utils import load_datasets
from rewards import thinking_reward_func, accuracy_reward_func, length_reward_func
from trainer import RLTrainer
from dotenv import load_dotenv
from pathlib import Path
import subprocess, os


@hydra.main(config_path="configs", version_base=None)
def main(cfg: RLModelTrainingConfig) -> None:

    cfg = OmegaConf.merge(OmegaConf.structured(RLModelTrainingConfig), cfg)

    # ------------------------------------------------------------------
    # Setup dataset
    # ------------------------------------------------------------------
    set_seed(cfg.seed)
    load_dotenv()
    raw_ds = load_datasets(cfg.dataset, cfg.seed)

    print(f"Loaded {len(raw_ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, trust_remote_code=True
    )

    def apply_chat_template(example):
        prompt, solution = example["prompt"], example["solution"]

        template = """
You are a math problem solver. Solve the following math problem and provide the solution in the boxed format.
It is crucial you start your solution with a <think> tag and end the thinking with a </think> tag after which you provide the solution in the boxed format.

Here is the math problem:
{problem}
"""
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": template.format(problem=prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "solution": solution}

    train_ds = raw_ds.map(apply_chat_template)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    reward_fns = [thinking_reward_func, accuracy_reward_func, length_reward_func]

    # ------------------------------------------------------------------
    # Train!
    # ------------------------------------------------------------------
    trainer = RLTrainer(cfg, train_dataset=train_ds, reward_functions=reward_fns)

    # We load last checkpoint if there is any.
    save_dir = cfg.checkpoint.save_dir
    if os.path.exists(save_dir) and os.listdir(save_dir):
        dirs = os.listdir(save_dir)
        dirs = list(sorted(dirs, key=lambda x: int(x.split("_")[-1])))
        last_checkpoint = os.path.join(save_dir, dirs[-1])
        trainer.load_checkpoint(last_checkpoint)

    result = trainer.train()
    if result:
        # trainer.push_to_hub()
        pass
    elif os.getenv("RANK") == "0":
        login_env_file = Path.home() / ".slurm_env" / "login_env.txt"

        clean_env = {}
        with login_env_file.open() as f:
            for line in f:
                if "=" in line:
                    try:
                        k, v = line.rstrip("\n").split("=", 1)
                        if not k.startswith("BASH_FUNC_"):
                            clean_env[k] = v
                    except ValueError:
                        pass

        for name in ("HOME", "USER", "LOGNAME"):
            clean_env[name] = os.environ.get(name, "")

        subprocess.run(
            ["sbatch", "--export=ALL", "grpo.sh"],
            env=clean_env,
            check=True,
        )


if __name__ == "__main__":
    main()
