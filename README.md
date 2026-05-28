<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### flash-attn Prebuilt Wheel

`flash-attn` builds CUDA kernels from source, slow and memory-intensive. Use a prebuilt wheel:

```bash
# 1. Check versions
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}')"

# 2. Download matching wheel from GitHub Releases
#    e.g. for torch 2.11+cu130 / cxx11abi=True / cp312:
curl -sL -o /tmp/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. Install and sync
uv pip install /tmp/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 4. Pin in pyproject.toml to prevent source rebuild
#    [tool.uv.sources]
#    flash-attn = { path = "/tmp/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" }

uv sync
python example.py
```

## Model Download

To download the model weights manually, use the following command:
```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)