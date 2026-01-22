#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>

#include <torch/types.h>
#include <cuda.h>
#include <ATen/Functions.h>

// RMSNorm kernel: y = (x / RMS(x)) * weight
// where RMS(x) = sqrt(mean(x^2) + eps)
__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int batch_size,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element
    
    if (row >= batch_size) return;
    
    const half* x = input + row * hidden_dim;
    const half* res = residual + row * hidden_dim;
    half* y = output + row * hidden_dim;
    
    // Step 1: Compute sum of squares using shared memory reduction
    __shared__ float shared_sum[32];  // For warp reduction
    
    float thread_sum = 0.0f;
    float x_new; // if this for loop happens more than once it will break, in this case we need to cache more than one x
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]) + ( __half2float(res[i]) * alpha);
        x_new = val;
        thread_sum += val * val;
    }
    
    // Warp-level reduction
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    float sum_sq = 0.0f;
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        sum_sq = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast RMS to all threads
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / hidden_dim;
        rms_shared = rsqrtf(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    }
    __syncthreads();
    
    float rms_inv = rms_shared;
    
    // Step 2: Normalize and apply weight
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float w = __half2float(weight[i]);
        y[i] = __float2half(x_new * rms_inv * w);
    }
}

void rmsnorm_cuda(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
) {
    cudaError_t result;
    
    int threads = K; // 512
    int blocks = MN;
    
    rmsnorm_kernel<<<blocks, threads>>>(
        (half *)input, (half *)residual, (half *)weight, (half *)output, MN, K, alpha, eps
    );

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "device synchronize failed: "
        << cudaGetErrorString(result) << std::endl;

        exit(1);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    auto MN = A.size(0) * A.size(1);
    auto K = A.size(2);

    float eps = 0.00001;
    float alpha = 2.44921875;

    auto o = torch::empty({A.size(0), A.size(1), K}, A.options());
    rmsnorm_cuda(
        (half *)A.data_ptr(),
        (half *)B.data_ptr(),
        (half *)C.data_ptr(),
        (half *)o.data_ptr(),
        MN,
        K,
        alpha,
        eps
    );

    return o;
}