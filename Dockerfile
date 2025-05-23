FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install build dependencies for GCC
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    flex \
    bison \
    libgmp-dev \
    libmpc-dev \
    libmpfr-dev \
    texinfo \
    libisl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install  \
    accelerate \
    antlr4-python3-runtime==4.9.3 \
    cachetools \
    colorama \
    dacite \
    datasets \
    deepspeed==0.15.4 \
    evaluate \
    fastapi \
    huggingface_hub \
    hydra-core==1.3.2 \
    ipywidgets \
    jinja2 \
    math-verify \
    multiprocess \
    numpy \
    omegaconf==2.3.0 \
    packaging \
    pandas \
    peft \
    polars \
    pydantic \
    PyYAML \
    safetensors \
    scikit-learn \
    seaborn \
    sentencepiece \
    sympy \
    tensorboard \
    timeout_decorator \
    tqdm \
    transformers \
    uvicorn \
    wandb \
    dotenv \
    trl \
    psutil

RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/f8d/08016585ac070/flash_attn-2.7.4.post1-cp312-cp312-linux_aarch64.whl#sha256=f8d08016585ac070056fcb3839b0b5f61a35b46d1f7d4132b9cef483d1c1d0aa
RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/d7d/a55d7cb5a6a84/triton-3.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=d7da55d7cb5a6a84b32435b0b7631c5f3b4676018a4c731b55da719c9d7708b1
RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/c55/1d0aa8532863b/torchvision-0.22.0-cp312-cp312-linux_aarch64.whl#sha256=c551d0aa8532863bc620a319bfcc2afe4650196528f52ce643e969b919335dd2
RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/59a/475eee6bfeae6/torch-2.7.0+cu128-cp312-cp312-linux_aarch64.whl#sha256=59a475eee6bfeae68b00af8f31fdb386107aad8795fe07310fd7c54a75ebd902
RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/08a/78cb3f1838b3c/vllm-0.8.4+cu128-cp312-cp312-linux_aarch64.whl#sha256=08a78cb3f1838b3c018525422cc3e167d9f9f7827b60f17a1869640dac6554a8

RUN pip install liger-kernel==0.5.8


RUN apt-get update && apt-get install -y build-essential cmake
RUN export BNB_CUDA_VERSION=128 && git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/ && cmake -DCOMPUTE_BACKEND=cuda -S . && make && pip install -e .


# Sanity check
RUN python -c "import torch, flash_attn, bitsandbytes, vllm, liger_kernel; print(torch.__version__)"

# Set working directory
RUN mkdir -p /workspace
WORKDIR /workspace
