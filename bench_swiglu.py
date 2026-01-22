import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os

repo_root = os.path.dirname(os.path.abspath(__file__))

# if the below code is hanging use this:  rm -r ~/.cache/torch_extensions/
# Load the CUDA kernel as a python module
cuda_func = load(
    name='cuda_func',
    sources=['main.cpp', 'fused_swiglu.cu'],
    extra_include_paths=[
        os.path.join(repo_root, 'cutlass', 'include'),
        os.path.join(repo_root, 'cutlass_ext', 'include'),
        os.path.join(repo_root, 'cutlass', 'examples', 'common'),
        os.path.join(repo_root, 'cutlass', 'tools', 'util', 'include'),
    ],
    extra_cuda_cflags=['-O2', '-use_fast_math'],
)

cuda_quant_gemm = load(
    name='cuda_quant_gemm',
    sources=['main.cpp', 'gemm_ampere.cu'],
    extra_include_paths=[
        os.path.join(repo_root, 'cutlass', 'include'),
        os.path.join(repo_root, 'cutlass_ext', 'include'),
        os.path.join(repo_root, 'cutlass', 'examples', 'common'),
        os.path.join(repo_root, 'cutlass', 'tools', 'util', 'include'),
    ],
    extra_cuda_cflags=['-O2', '-use_fast_math'],
)

cuda_silu = load(
    name='cuda_silu',
    sources=['main_silu.cpp', 'silu_mul.cu'],
    extra_cuda_cflags=['-O2', '-use_fast_math'],
)

@torch.inference_mode()
def gemm_fused(
    x,
    w
):
    return cuda_func.forward(x, w)

@torch.inference_mode()
def gemm_ref(
    x,
    w
):
    t = x @ w.t()
    chunks = t.chunk(2, -1)
    y = chunks[0]
    gate = chunks[1]
    return torch.nn.functional.silu(gate).mul(y)

@torch.inference_mode()
def gemm_cuda(
    x,
    w
):
    t = cuda_quant_gemm.forward(x, w)
    return cuda_silu.forward(t)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 800
timestep = 833
out_features = 2048 * 2
in_features = 512

A = torch.randn(batch_size, timestep, in_features).cuda().half() # input
B = torch.randn(out_features, in_features).cuda().half() # weights

print('=== profiling python func ===')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    C = gemm_fused(A, B)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== cuda func === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    C_cuda = gemm_cuda(A, B)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(C.size())
print(C_cuda.size())

print(C)
print(C_cuda)

print('values sanity check:', torch.allclose(C, C_cuda, atol=1e-0))