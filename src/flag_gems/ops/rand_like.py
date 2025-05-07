import logging

import torch
import triton

from flag_gems.ops.rand import rand_kernel
from flag_gems.runtime import torch_device_fn, get_torch_device_ctx
from flag_gems.utils.random_utils import philox_backend_seed_offset

UNROLL = 4


def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    if device.type == 'cpu':
        # OPTIM:
        philox_seed, philox_offset = torch.seed(), 0
    else:
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with get_torch_device_ctx(x.device):
        rand_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out
