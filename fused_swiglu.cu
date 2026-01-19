#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Functions.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#include "dual_gemm/device/dual_gemm.h"
#include "swiglu_kernel.h"

template <typename T>
using SiLu = cutlass::epilogue::thread::SiLu<T>;

template <typename scalar_t, template <typename> typename ActivationFn>
static void dual_gemm_lhs_activation_and_mul_cuda(
    void *x,
    void *w0,
    void *w1,
    void *d0,
    void *d1,
    void *d2, // result
    int64_t B,
    int64_t I,
    int64_t H
) {
    cudaError_t result;
    
    int d_stride_0 = H;
    int x_stride_0 = I;
    int w_stride_0 = I;

    // templati-ze the cutlass kernel
    cutlass::gemm::GemmCoord problem_size(B, H, I);

    constexpr int kStages = 3;
    constexpr bool kSplitKSerial = false;

    using ElementOutput = scalar_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
    using EpilogueOutputOp2 = EpilogueLHSActivationAndMul<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ActivationFn,
        ElementOutput,
        ElementCompute>;

    const ElementCompute alpha0 = ElementCompute(1);
    const ElementCompute beta0 = ElementCompute(0);
    const ElementCompute alpha1 = ElementCompute(1);
    const ElementCompute beta1 = ElementCompute(0);

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // Optionally, we might not need intermediate GEMM outputs
    constexpr bool kStoreD0 = true;
    constexpr bool kStoreD1 = true;
    using ArchTag = cutlass::arch::Sm80;

    using DualGemm = cutlass::gemm::device::DualGemm<
        scalar_t,
        cutlass::layout::RowMajor,
        scalar_t,
        cutlass::layout::ColumnMajor,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp01,
        EpilogueOutputOp01,
        EpilogueOutputOp2,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
        kStages,
        kStoreD0,
        kStoreD1,
        kSplitKSerial>;
    // {
    //     cudaDeviceProp *p = getDeviceProperties(x.device().index());
    //     ASSERT(p->major * 10 + p->minor >= ArchTag::kMinComputeCapability)
    // }

    int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
    using RefA = typename cutlass::TensorRef<typename DualGemm::ElementA, typename DualGemm::LayoutA>;
    using RefB0 = typename cutlass::TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB0>;
    using RefB1 = typename cutlass::TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB1>;
    using RefC = typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;
    RefC ref_b0, ref_b1;

    typename DualGemm::Arguments arguments{
        cutlass::gemm::DualGemmMode::kGemm,
        problem_size,
        RefA{
            (scalar_t *)x,
            typename DualGemm::LayoutA::Stride(x_stride_0)},
        RefB0{
            (scalar_t *)w0,
            typename DualGemm::LayoutB0::Stride(w_stride_0)},
        ref_b0,
        RefC{
            (scalar_t *)d0,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        RefB1{
            (scalar_t *)w1,
            typename DualGemm::LayoutB1::Stride(w_stride_0)},
        ref_b1,
        RefC{
            (scalar_t *)d1,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        RefC{
            (scalar_t *)d2,
            typename DualGemm::LayoutC::Stride(d_stride_0)},
        typename DualGemm::EpilogueOutputOp0::Params{alpha0, beta0},
        typename DualGemm::EpilogueOutputOp1::Params{alpha1, beta1},
        typename DualGemm::EpilogueOutputOp2::Params{},
        split_k_slices};

    DualGemm dual_gemm;

    uint8_t *workspace;
    cudaMalloc((void **)&workspace, sizeof(uint8_t) * dual_gemm.get_workspace_size(arguments));

    cutlass::Status status = dual_gemm.can_implement(arguments);
    assert(status == cutlass::Status::kSuccess);

    status = dual_gemm.initialize(arguments, workspace);
    assert(status == cutlass::Status::kSuccess);

    status = dual_gemm();
    assert(status == cutlass::Status::kSuccess);

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "device synchronize failed: "
        << cudaGetErrorString(result) << std::endl;

        exit(1);
    }

    result = cudaFree(workspace);
	if (result != cudaSuccess) {
        std::cerr << "device synchronize failed: "
        << cudaGetErrorString(result) << std::endl;

        exit(1);
    }
}

torch::Tensor forward(torch::Tensor x, torch::Tensor w) {
    int64_t B = x.size(0) * x.size(1); // batch * timestep size
    int64_t I = w.size(1); // input dim or hidden dim
    int64_t H = w.size(0) / 2; // output dim

    auto d0 = torch::zeros(B * H, x.options());
    auto d1 = torch::zeros(B * H, x.options());
    auto d2 = torch::zeros({x.size(0), x.size(1), H}, x.options());

    auto weights = w.chunk(2, 0);

    dual_gemm_lhs_activation_and_mul_cuda<cutlass::half_t, SiLu>(
        x.data_ptr(),
        weights[0].data_ptr(), weights[1].data_ptr(),
        d0.data_ptr(), d1.data_ptr(), d2.data_ptr(),
        B, I, H
    );

    return d2;
}