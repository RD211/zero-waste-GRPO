import gc
import torch
import random
import numpy as np
import ctypes
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor


def get_shareable_version(meta):
    return {
        key: {k: v for k, v in meta[key].items() if k != "_shm_obj"} for key in meta
    }


def create_shared_state_dict(state_dict, max_workers=200):

    def make_shm(key, tensor):
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()

        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()

        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        buf = shm.buf

        if dtype == torch.bfloat16:
            # one raw-bytes copy from the tensors data_ptr
            ptr = tensor.data_ptr()
            raw = ctypes.string_at(ptr, nbytes)
            buf[:nbytes] = raw
        else:
            # fallback via numpy
            np_arr = tensor.numpy()
            shm_arr = np.ndarray(shape, dtype=np_arr.dtype, buffer=buf)
            shm_arr[:] = np_arr

        del tensor
        if random.random() < 0.02:
            torch.cuda.empty_cache()
            gc.collect()

        return key, {
            "shm_name": shm.name,
            "shape": shape,
            "dtype": str(dtype),
            "nbytes": nbytes,
            "_shm_obj": shm,
        }

    shared_meta = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_shm, k, v) for k, v in state_dict.items()]
        for f in futures:
            k, m = f.result()
            shared_meta[k] = m
    return shared_meta


def load_shared_state_dict(meta):

    def load_shared_tensor(key, info):
        shm = shared_memory.SharedMemory(name=info["shm_name"])
        shape = tuple(info["shape"])
        dtype = info["dtype"]
        nbytes = info["nbytes"]
        try:
            if "bfloat16" in dtype or dtype.endswith("bf16"):
                # copy raw bytes out via a Python bytes object
                buf = shm.buf
                raw = buf[:nbytes].tobytes()
                tensor = torch.empty(shape, dtype=torch.bfloat16)
                ctypes.memmove(tensor.data_ptr(), raw, nbytes)
            else:
                buf = shm.buf
                np_arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=buf)
                tensor = torch.tensor(np_arr)
                del np_arr
            if random.random() < 0.1:
                gc.collect()
            return key, tensor
        finally:
            shm.close()
            shm.unlink()

    state_dict = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(load_shared_tensor, k, info) for k, info in meta.items()
        ]
        for f in futures:
            k, t = f.result()
            state_dict[k] = t
    return state_dict
