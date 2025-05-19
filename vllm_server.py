import os
import hydra
import uvicorn

from omegaconf import OmegaConf
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from configs.cfg import RLModelTrainingConfig
from shared_memory import load_shared_state_dict
from hydra.core.config_store import ConfigStore
from vllm import SamplingParams
from vllm_inference import SimplevLLMInference
from slurm_utils import *
from dotenv import load_dotenv
from utils import init_logger

load_dotenv()


logger = init_logger()
cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

config: RLModelTrainingConfig = None
app = FastAPI()
vllm_inference = None
sampling_params = None


class GenerateRequest(BaseModel):
    problems: List[str]
    meta: dict = {}


@app.post("/generate")
def generate(request: GenerateRequest):
    global vllm_inference
    logger.info(f"Received {len(request.problems)} problems")
    state_dict = load_shared_state_dict(request.meta)
    logger.info(f"Loaded {len(state_dict)} tensors from shared memory")
    # If any keys start with "module.", remove them
    state_dict = {
        k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")
    }

    generations = vllm_inference.run_batch(
        messages=request.problems,
        sampling_params=sampling_params,
        state_dict=state_dict,
    )
    # We get only the text
    generations = [g.outputs[0].text for g in generations]
    return {"generations": generations}


@app.post("/shutdown")
def shutdown():
    global vllm_inference
    vllm_inference.cleanup()
    os._exit(0)
    return {"status": "shutdown"}


@hydra.main(config_path="configs", version_base=None)
def main(cfg: RLModelTrainingConfig):
    global vllm_inference, config, sampling_params

    default_config = OmegaConf.structured(RLModelTrainingConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    config = cfg

    model_config = cfg.model
    vllm_inference = SimplevLLMInference(
        model_path=model_config.model_name_or_path,
        gpus_per_instance=model_config.number_of_gpus_per_instance,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        max_model_len=model_config.max_length * 2,
        max_num_seqs=model_config.max_num_seqs,
        enforce_eager=model_config.enforce_eager,
        use_v0=model_config.use_v0,
    )
    sampling_params = SamplingParams(
        temperature=model_config.temperature,
        top_k=model_config.top_k,
        top_p=model_config.top_p,
        max_tokens=model_config.max_length,
        min_tokens=model_config.max_length,
    )

    uvicorn.run(app, host="0.0.0.0", port=cfg.server_port)


if __name__ == "__main__":
    main()
