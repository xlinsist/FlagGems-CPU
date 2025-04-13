# FlagGems-CPU

## Introduction

This is a CPU development branch of FlagGems. We are temporarily naming this fork "FlagGems-CPU".

## Quick Start

### Requirements

1. PyTorch >= 2.2.0
2. Transformers >= 4.40.2
3. Triton-CPU

**Note:** Triton-CPU is currently experimental and need to be installed from source. Refer to [the official repo](https://github.com/triton-lang/triton-cpu) for installation.

### Installation

```shell
cd <your-work-dir>
git clone git@github.com:xlinsist/FlagGems-CPU.git
cd FlagGems
git checkout cpu-dev # We implemented CPU-specific changes here
pip install .
```

## Usage

1. Verify successful installation by running the following Python code:
```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device="cpu")
B = torch.randn((K, N), dtype=torch.float16, device="cpu")
with flag_gems.use_gems():
    C1 = torch.mm(A, B)
print(C1)
C2 = torch.mm(A, B)
print()
print(C2)
```

2. Benchmark operator performance. Take mm and layer_norm for example:
```shell
cd benchmark
time pytest test_blas_perf.py -s --mode cpu --record log --level core --dtypes float32 --warmup 25 --iter 100 -m mm
time pytest test_norm_perf.py -s --mode cpu --record log --level core --dtypes float32 --warmup 25 --iter 100 -m layer_norm
```
