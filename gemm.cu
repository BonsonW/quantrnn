#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Functions.h>

struct ScaledTensor {
    at::Tensor values;  // int8 tensor
    at::Tensor scale;   // float scale per slice
};

ScaledTensor quantize_tensor(const at::Tensor& t, int dim) {
    auto fp_range = t.abs().amax(dim);
    constexpr int levels = 256;
    auto quant_scale = ((levels / 2) / fp_range);
    auto quant_max = (levels / 2) - 1;
    auto t_quant = (t * quant_scale.unsqueeze(dim)).round().clip(-quant_max, quant_max);
    return ScaledTensor{t_quant.to(at::ScalarType::Char), quant_scale.to(at::ScalarType::Float).reciprocal_()};
}

// row major
__global__
void sgemm_naive(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C
) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0) * A.size(1); // batch * timestep size
    int K = A.size(-1); // in features
    int N = B.size(-1); // out features

    torch::Tensor C = torch::empty({A.size(0), A.size(1), N}).cuda();

    A.contiguous();
    B.contiguous();
    C.contiguous();

    float alpha = 1.0;
    float beta = 0.0;

    // create as many blocks as necessary to map all of C
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32), 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, (float *)A.data_ptr(), (float *)B.data_ptr(), beta, (float *)C.data_ptr());

    return C;
}