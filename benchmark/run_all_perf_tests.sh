#!/bin/bash

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

LOG_FILE="log_run_all.txt"
> "$LOG_FILE"

all_benchmark_names=("attention" "binary_pointwise" "blas" "distribution" "fused" "generic_pointwise" "norm" "reduction" "select_and_slice" "special" "tensor_concat" "tensor_constructor" "unary_pointwise")
# "attention", "fused", "binary_pointwise", "unary_pointwise" are removed for not fully supporting float32.
legal_benchmark_names=("blas" "distribution" "generic_pointwise" "norm"  "reduction" "select_and_slice" "special" "tensor_concat" "tensor_constructor")

# take "blas" and "norm" for example
benchmark_names=("blas" "norm")

for name in "${benchmark_names[@]}"; do
    {
        echo "Running test: $name with 16 threads"
        time pytest "test_${name}_perf.py" -s \
            --mode cpu \
            --record log \
            --level core \
            --dtypes float32 \
            --warmup 25 \
            --iter 100
        echo "---------------------------------------------"
    } >> "$LOG_FILE" 2>&1
done

OUTPUT_FILE="result_test_all.log"
> "$OUTPUT_FILE"

for log_file in result_test_*--warmup_25--iter_100.log; do
    if [[ -f "$log_file" ]]; then
        cat "$log_file" >> "$OUTPUT_FILE"
    fi
done
