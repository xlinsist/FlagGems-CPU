import torch
import flag_gems
import pytest
import numpy as np
import random

from .conftest import TO_CPU
from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE


# M, N, K = 1024, 1024, 1024
# A = torch.randn((M, K), dtype=torch.float32, device="cpu")
# B = torch.randn((K, N), dtype=torch.float32, device="cpu")
#
# def test_mm():
#     with flag_gems.use_gems():
#         result = torch.mm(A, B)
#     expect = torch.mm(A, B)
#     torch.testing.assert_close(result, expect, rtol=1e-4, atol=1e-4)
#
# def test_layernorm():
#     normalized_shape = (N,)
#     eps = 1e-5
#
#     with flag_gems.use_gems():
#         result = torch.nn.functional.layer_norm(A, normalized_shape, eps=eps)
#
#     expect = torch.nn.functional.layer_norm(A, normalized_shape, eps=eps)
#
#     torch.testing.assert_close(result, expect, rtol=1e-4, atol=1e-4)

# @pytest.mark.parametrize("shape", POINTWISE_SHAPES)
# @pytest.mark.parametrize("float_type", [torch.float32])
# def test_type_promotion_int_to_float(shape, float_type):
#     # arg0:float
#     inp_float = torch.randn(shape, dtype=float_type, device=flag_gems.device)
#     ref_inp = to_reference(inp_float, False)
#     ref_out = torch.sin(ref_inp)
#     with flag_gems.use_gems():
#         res_out = torch.sin(inp_float)
#     gems_assert_close(res_out, ref_out, float_type)
#
#     # arg0:int
#     # inp_int = torch.randint(10, shape, device=flag_gems.device)
#     # ref_inp_int = to_reference(inp_int, True)
#     # ref_out = torch.sin(ref_inp_int)
#     # with flag_gems.use_gems():
#     #     res_out = torch.sin(inp_int)
#     # gems_assert_close(res_out, ref_out, torch.float32)
#

MN_SHAPES = [(1, 32)] if QUICK_MODE else [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = [(9999, 9999, 9999)]
# MNK_SHAPES = (
#     [(1, 1, 32)] if QUICK_MODE else [(1, 1, 32), (15, 160, 1024), (495, 5333, 71), (9999, 9999, 9999)]
# )
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
SCALARS = [0.001, -0.999, 100.001, -111.999]
INT_DTYPES = [torch.int16, torch.int32]
fp64_is_supported = flag_gems.runtime.device.support_fp64
bf16_is_supported = flag_gems.runtime.device.support_bf16
int64_is_supported = flag_gems.runtime.device.support_int64
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64] if fp64_is_supported else FLOAT_DTYPES
ALL_INT_DTYPES = INT_DTYPES + [torch.int64] if int64_is_supported else INT_DTYPES
DISTRIBUTION_SHAPES = [(20, 320, 15)]
KRON_SHAPES = [[(), (2, 3)]]
BOOL_TYPES = [torch.bool]
CAMBRICON_STACK_SHAPES = [
    [
        (8, 8, 128),
        (8, 8, 128),
        (8, 8, 128),
    ],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]
STACK_SHAPES = [[(4,), (4,)]]
STACK_SHAPES_TEST = STACK_SHAPES + (
    CAMBRICON_STACK_SHAPES if flag_gems.vendor_name == "cambricon" else []
)
STACK_DIM_LIST = [-2, -1, 0, 1]
REDUCTION_SHAPES = [(96, 32)] 
# CUMSUM_SHAPES = ([(2, 32)])
CUMSUM_SHAPES = (
    REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)

# @pytest.mark.mm
# @pytest.mark.parametrize("M, N, K", MNK_SHAPES)
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# def test_accuracy_mm(M, N, K, dtype):
#     mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
#     mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
#     ref_mat1 = to_reference(mat1, True)
#     ref_mat2 = to_reference(mat2, True)
#
#     ref_out = torch.mm(ref_mat1, ref_mat2)
#     with flag_gems.use_gems():
#         res_out = torch.mm(mat1, mat2)
#
#     gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

@pytest.mark.cumsum
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.cumsum(inp, dim=dim)
    else:
        with flag_gems.use_gems():
            res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
