#include <math.h>
#include <float.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Functions.h>

__global__ void silu_mul(
	half *x_gpu,
	half *o_gpu,
    const uint64_t B,
    const uint64_t M
) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && j < M) {
        uint64_t k = i + j * B;
        i += j * (B * 2);

        half y = x_gpu[i];
        half gate = x_gpu[i + B];

        float g = __half2float(gate);
        float silu = g / (1.0f + __expf(-g));

        o_gpu[k] = __float2half(silu * __half2float(y));
    }
}

void silu_mul_cuda(
    void *x_gpu,
    void *o_gpu,
    uint64_t M,
    uint64_t B
) {
    cudaError_t result;

    dim3 block(32, 32);
	dim3 grid(
        (B + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    silu_mul<<<grid, block>>>(
        (half *)x_gpu,
        (half *)o_gpu,
        B,
        M
    );

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "cuda kernel failed: " << std::endl;
        exit(1);
    }
}

torch::Tensor forward(torch::Tensor A) {
    auto M = A.size(0) * A.size(1);
    auto B = A.size(-1) / 2;

    auto o = torch::empty({A.size(0), A.size(1), B}, A.options());
    silu_mul_cuda(
        A.data_ptr(),
        o.data_ptr(),
        M,
        B
    );
    return o;
}