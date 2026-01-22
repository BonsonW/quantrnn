import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os
import time


repo_root = os.path.dirname(os.path.abspath(__file__))

# Load the CUDA kernel as a python module
# cuda_gemm = load(name='cuda_gemm', sources=['main.cpp', 'gemm.cu'], extra_cuda_cflags=['-O2', '-use_fast_math'])
cuda_gemm = load(
    name='cuda_gemm',
    sources=['main_rms.cpp', 'rms_norm.cu'],
    extra_include_paths=[
        os.path.join(repo_root, 'cutlass', 'include'),
        # os.path.join(repo_root, 'cutlass_ext', 'include'),
        # os.path.join(repo_root, 'cutlass', 'examples', 'common'),
        # os.path.join(repo_root, 'cutlass', 'tools', 'util', 'include'),
    ],
    extra_cuda_cflags=['-O2', '-use_fast_math'],
)

@torch.inference_mode()
def gemm_ref(
    A,
    B
):
    return A @ B

# Use small model params, otherwise slower than manual attention. See caveats in README.
# batch_size = 512
# timestep = 1024
# out_features = 512

batch_size = 1
timestep = 1
out_features = 32

x = torch.randn(batch_size * timestep * out_features).resize(batch_size, timestep, out_features).half().cuda().contiguous() # input
inp = torch.randn(batch_size * timestep * out_features).resize(batch_size, timestep, out_features).half().cuda().contiguous() # input
w = torch.randn(out_features).resize(out_features).half().cuda().contiguous() # weights

print('=== cuda gemm === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    D_cuda = cuda_gemm.forward(inp, x, w)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling python gemm ===')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    alpha = 2.44921875
    k = inp + (x * alpha)
    D = torch.nn.functional.rms_norm(k, (inp.size(2),), weight=w, eps=0.00001)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(D.size())
print(D_cuda.size())

print(D)
print(D_cuda)

print('values sanity check:', torch.allclose(D, D_cuda, atol=1e-02))