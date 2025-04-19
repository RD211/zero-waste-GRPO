import numpy as np
import torch
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor

def create_shared_tensor(key, tensor):
    try:
        # Convert bfloat16 to float32 because NumPy doesn't support bfloat16.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        np_array = tensor.cpu().numpy()

        # Create a shared memory block and copy the array into it.
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        shared_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
        shared_array[:] = np_array[:]

        # Return the metadata for this tensor.
        meta = {
            'shm_name': shm.name,
            'shape': np_array.shape,
            'dtype': np_array.dtype.str,
            'nbytes': np_array.nbytes,
            '_shm_obj': shm  # Keep reference to avoid garbage collection.
        }
        return key, meta
    except Exception as e:
        print(e)
        raise e

def create_shared_state_dict(state_dict):
    shared_meta = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_shared_tensor, key, tensor)
                   for key, tensor in state_dict.items()]
        for future in futures:
            key, meta = future.result()
            shared_meta[key] = meta
    return shared_meta

def get_shareable_version(meta):
    return {
        key: {k: v for k, v in meta[key].items() if k != '_shm_obj'}
        for key in meta
    }

def load_shared_tensor(key, info):
    shm = shared_memory.SharedMemory(name=info['shm_name'])
    try:
        np_array = np.ndarray(info['shape'], dtype=np.dtype(info['dtype']), buffer=shm.buf)
        tensor = torch.tensor(np_array)
    finally:
        shm.close()
    return key, tensor

def load_shared_state_dict(meta):
    state_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(load_shared_tensor, key, info)
                   for key, info in meta.items()]
        for future in futures:
            key, tensor = future.result()
            state_dict[key] = tensor
    return state_dict
