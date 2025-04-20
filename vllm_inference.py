import gc
import math
import torch
from utils import init_logger
from vllm import LLM

class SimplevLLMInference:
    def __init__(
        self,
        model_path,
        gpus_per_instance=4,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_seqs=512,
        enforce_eager=True,
        use_v0=False,
    ):
        self.model_path = model_path
        self.gpus_per_instance = gpus_per_instance
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.use_v0 = use_v0

        if self.use_v0:
            import os
            os.environ["VLLM_USE_V1"] = "0"
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.logger = init_logger(rank=os.getenv("LOCAL_RANK", 0))
        # Initialize the model
        self.logger.info(f"Loading model from {self.model_path} on {self.gpus_per_instance} GPUs")

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.gpus_per_instance,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=True,
            enable_sleep_mode=True,
        )

        self.llm.sleep()
        self.logger.info("Model loaded successfully")

    def run_batch(self, messages, sampling_params, state_dict=None):
        self.logger.info("Waking up the model for inference")
        self.llm.wake_up()
        self.logger.info("Model is awake")

        prompts = messages

        # Load weights if provided
        self.logger.info("Loading state dict")
        if state_dict:
            self.logger.info(f"Loading state dict with {state_dict.keys()}")
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                state_dict.items()
            )
        self.logger.info("State dict loaded")
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        self.logger.info("Inference completed")

        # Memory cleanup
        self.llm.sleep()
        gc.collect()
        torch.cuda.empty_cache()

        self.logger.info("Model is asleep")

        return outputs

    def cleanup(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        self.logger.info("vllmInference cleaned up")


