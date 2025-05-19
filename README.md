# FlagGems-CPU

## Introduction

This is a CPU development branch of FlagGems. We are temporarily naming this fork "FlagGems-CPU". Please refer to [CPUPorting.md](https://github.com/xlinsist/FlagGems-CPU/blob/cpu-dev/CPUPorting.md) for the key modifications and compatibility range on CPU porting.

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

2. Benchmark single operator performance. Take `mm` and `layer_norm` for example:
```shell
cd benchmark
time pytest test_blas_perf.py -s --mode cpu --record log --level core --dtypes float32 --warmup 25 --iter 100 -m mm
time pytest test_norm_perf.py -s --mode cpu --record log --level core --dtypes float32 --warmup 25 --iter 100 -m layer_norm
```
> FIXME: Resolve the error when running scripts in the benchmark directory due to `performance_utils.py` being incompatible with CPU. Specifically, the line with "torch_backend_device.matmul.allow_tf32 = False" fails with: "AttributeError: module 'torch.backends.cpu' has no attribute 'matmul'".


3. Benchmark all ops performance. You can modify `run_all_perf_tests.sh` to test the ops you want.
```shell
cd benchmark
bash run_all_perf_tests.sh # It will generate `result_test_all.log`
python summary_for_plot.py result_test_all.log
```
It will generate `result_test_summary.log` with contents like this:
```
op_name                        float32_speedup      all_tests_passed    
addmm                          1.215471             yes                 
batch_norm                     1.020813             yes                 
bmm                            1.186168             yes                 
group_norm                     1.048566             yes                 
instance_norm                  1.013160             yes                 
layer_norm                     0.994558             yes                 
mm                             1.218488             yes                 
mv                             1.028544             yes                 
outer                          0.982253             yes                 
vdot                           1.020879             yes                 
vector_norm                    1.094137             yes                 
weight_norm                    0.981210             yes                 
weight_norm_interface          0.967518             yes                 
```

4. Compare all ops performance.
```shell
cd benchmark
python summary_for_plot.py result_test_all_your_methmod.log -c result_test_all_baseline_methmod.log
```
It will generate `result_test_compare.log` with contents like this:
```
op_name                        float32_speedup      comp_fp32_speedup   all_tests_passed    
addmm                          1.215471             0.929496            yes                 
batch_norm                     1.020813             0.983629            yes                 
bmm                            1.186168             1.014502            yes                 
group_norm                     1.048566             0.895594            yes                 
instance_norm                  1.013160             0.962255            yes                 
layer_norm                     0.994558             1.028554            yes                 
mm                             1.218488             1.035678            yes                 
mv                             1.028544             0.924285            yes                 
outer                          0.982253             0.894619            yes                 
vdot                           1.020879             0.940504            yes                 
vector_norm                    1.094137             1.062247            yes                 
weight_norm                    0.981210             0.967043            yes                 
weight_norm_interface          0.967518             0.920457            yes                 
```
