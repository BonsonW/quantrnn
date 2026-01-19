import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
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
    A,
    B
):
    return torch.matmul(A, B)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
timestep = 8
out_features = 8
in_features = 4

A = torch.arange(batch_size * timestep * in_features).resize(batch_size, timestep, in_features).float().cuda() # input
B = torch.arange(out_features * in_features).resize(out_features, in_features).float().cuda() # weights

print(A.size())

print('=== cuda gemm === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    # C_cuda = cuda_gemm.forward(A, B)
    C_cuda = cuda_gemm.forward(B.t().contiguous(), A).resize(batch_size, timestep, out_features) # transposing because cutlass expects column major
    # C_cuda = cuda_gemm.forward(A, B).resize(batch_size, timestep, out_features) # transposing because cutlass expects column major
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling python gemm ===')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    C = gemm_ref(A, B.t())
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(C.size())
print(C_cuda.size())

print(C)
print(C_cuda)

print('values sanity check:', torch.allclose(C, C_cuda, atol=1e-05))