import math

import torch
import tensorhue
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from einops import rearrange, repeat
import os

repo_root = os.path.dirname(os.path.abspath(__file__))

# Load the CUDA kernel as a python module
# cuda_gemm = load(name='cuda_gemm', sources=['main.cpp', 'gemm.cu'], extra_cuda_cflags=['-O2', '-use_fast_math'])
cuda_gemm = load(
    name='cuda_gemm',
    sources=['main.cpp', 'gemm_cutlass.cu'],
    extra_include_paths=[
        os.path.join(repo_root, 'cutlass', 'include'),
        os.path.join(repo_root, 'cutlass', 'examples', 'common'),
        os.path.join(repo_root, 'cutlass', 'tools', 'util', 'include'),
    ],
    extra_cuda_cflags=['-O2', '-use_fast_math'],
)

@torch.inference_mode()
def gemm_ref(
    a,
    b
):
    return a @ b

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 128
out_features = 128
in_features = 64

A = torch.randn(batch_size, in_features).cuda() # input
B = torch.randn(in_features, out_features).cuda() # weights

print('=== profiling python gemm ===')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    C = gemm_ref(A, B)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== cuda gemm === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    C_cuda = cuda_gemm.forward(A, B)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(C.size())
print(C_cuda.size())

print('values sanity check:', torch.allclose(C, C_cuda))
