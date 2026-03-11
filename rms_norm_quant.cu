#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>

#include <torch/types.h>
#include <cuda.h>
#include <ATen/Functions.h>

struct tensor_quant {
    at::Tensor tensor; // int8 tensor
    at::Tensor scale;  // float scale per slice, reciprocal pre-applied
};

inline tensor_quant quantize_tensor(const at::Tensor &x, int dim) {
    auto fp_range = x.abs().amax(dim);
    auto i_range = 256 / 2;
    auto quant_scale = (i_range / fp_range);
    auto quant_max = i_range - 1;
    auto x_quant = (x * quant_scale.unsqueeze(dim)).round().clip(-quant_max, quant_max);
    return tensor_quant {
        x_quant.to(torch::kInt8).contiguous(),
        quant_scale.to(torch::kFloat32).reciprocal_().contiguous()
    };
}

// RMSNorm kernel: y = (x / RMS(x)) * weight
// where RMS(x) = sqrt(mean(x^2) + eps)
__global__ void rmsnorm_kernel(
    const half* input,
    const half* weight,
    int8_t* residual,
    float* residual_scale,
    int batch_size,
    int hidden_dim,
    float alpha,
    float eps
) {
    int row = blockIdx.x;  // Which sequence/batch element
    int idx = threadIdx.x;
    assert(idx < hidden_dim);
    
    if (row >= batch_size) return;
    
    const half* inp = input + row * hidden_dim;
    int8_t* res = residual + row * hidden_dim;
    float* res_scale = residual_scale + row;
    float w = __half2float(weight[idx]);
    
    // Step 1: Compute sum of squares using shared memory reduction
    __shared__ float shared_sum[32];  // For warp reduction
    
    float thread_sum = 0.0f;
    float val = __half2float(inp[idx]) + (((float)res[idx] * (*res_scale)) * alpha);
    thread_sum += val * val;
    
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

    // Step 2: Find max absolute value for output quantization
    __shared__ float shared_max[32];
    
    float thread_max = 0.0f;
    float normalized = val * rms_inv * w;
    thread_max = fmaxf(thread_max, fabsf(normalized));
    
    // Reduce to find max
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    float abs_max = 0.0f;
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        abs_max = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            abs_max = fmaxf(abs_max, __shfl_down_sync(0xffffffff, abs_max, offset));
        }
    }
    
    // write to quant scale
    __shared__ float quant_scale_shared;
    if (threadIdx.x == 0) {
        quant_scale_shared = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
        *res_scale = 1.0f / quant_scale_shared;
    }
    __syncthreads();
    
    
    // clamp and write quantized norm
    float quant_scale = quant_scale_shared;
    int quantized = __float2int_rn(normalized * quant_scale);
    quantized = max(-127, min(127, quantized));
    res[idx] = (int8_t)quantized;
}

void rmsnorm_cuda(
    const void* input,
    const void* weight,
    const void* residual,
    const void* residual_scale,
    int MN,
    int K,
    float alpha,
    float eps
) {
    cudaError_t result;
    assert(K <= 1024);
    
    int threads = K;
    int blocks = MN;
    
    rmsnorm_kernel<<<blocks, threads>>>(
        (half *)input, (half *)weight, (int8_t *)residual, (float *)residual_scale, MN, K, alpha, eps
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

    auto b_quant = quantize_tensor(B, -1);

    float eps = 0.00001;
    float alpha = 2.44921875;

    rmsnorm_cuda(
        A.data_ptr(),
        C.data_ptr(),
        b_quant.tensor.data_ptr(),
        b_quant.scale.data_ptr(),
        MN,
        K,
        alpha,
        eps
    );

    return (b_quant.tensor.to(at::kFloat) * b_quant.scale.unsqueeze(-1)).to(at::kHalf);
}