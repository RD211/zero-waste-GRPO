import os
import gc
import torch
from src.utils.utils import init_logger
from vllm import LLM


def echo_model_runner(self):
    return self.model_runner.model.__class__


def get_state_buffers(self):
    named_buffers = dict(self.model_runner.model.named_buffers())
    state_dict = {k: v.clone() for k, v in named_buffers.items()}
    # We move to cpu
    for k, v in state_dict.items():
        if v.device.type == "cuda":
            state_dict[k] = v.cpu()
    return state_dict


def set_state_buffers(self, state_dict):
    named_buffers = dict(self.model_runner.model.named_buffers())
    for k, v in state_dict.items():
        if k in named_buffers:
            if v.device.type == "cuda":
                named_buffers[k].copy_(v.cuda())
            else:
                named_buffers[k].copy_(v)
        else:
            raise KeyError(f"Key {k} not found in model buffers")


def load_weights(self, state_dict):
    self.model_runer.model.load_weights(state_dict.items())


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
            os.environ["VLLM_USE_V1"] = "0"
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            # https://github.com/vllm-project/vllm/issues/12774
            # os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        self.logger = init_logger(rank=os.getenv("LOCAL_RANK", 0))
        # Initialize the model
        self.logger.info(
            f"Loading model from {self.model_path} on {self.gpus_per_instance} GPUs"
        )

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
        self.logger.info("Model loaded successfully")

        # Hacky workaround for level=2 offload. fixed a very recent github commit but I am a bit lazy to build vllm.
        if self.use_v0:
            self.buffers = dict(
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_buffers()
            )
            self.buffers = {k: v.clone() for k, v in self.buffers.items()}
        else:
            self.buffers = self.llm.collective_rpc(get_state_buffers)[0]

        self.llm.sleep(1)

    def run_batch(self, messages, sampling_params, state_dict=None):
        self.logger.info("Waking up the model for inference")
        self.llm.wake_up()
        if self.use_v0:
            empty_buffers = dict(
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_buffers()
            )
            for k, v in empty_buffers.items():
                v.copy_(self.buffers[k])
        else:
            self.llm.collective_rpc(lambda x: set_state_buffers(x, self.buffers))
        self.logger.info("Model is awake")

        prompts = messages

        # Load weights if provided
        self.logger.info("Loading state dict")
        if state_dict:
            self.logger.info(f"Loading state dict with {state_dict.keys()}")
            if self.use_v0:
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                    state_dict.items()
                )
            else:
                self.llm.collective_rpc(lambda x: load_weights(x, state_dict.items()))

        self.logger.info("State dict loaded")

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        self.logger.info("Inference completed")

        # Memory cleanup
        self.llm.sleep(1)
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
