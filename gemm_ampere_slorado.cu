/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include "cutlass_ext/gemm_universal_base_compat.h"
#include "cutlass_ext/epilogue_per_row_per_col.h"
#include "cutlass_ext/gemm_with_epilogue_visitor.h"

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Functions.h>

#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

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
        x_quant.to(torch::kInt8),
        quant_scale.to(torch::kFloat32).reciprocal_()
    };
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInput = int8_t;                        // <- data type of elements in input matrix
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D
using ElementCompute = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using OperatorClass = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
// using ShapeMMAThreadBlock =
//     cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// // This code section describes tile size a warp will compute
// using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// // This code section describes the size of MMA op
// // using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16, for int8

#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s \n in file: %s, line number: %d\n", cudaGetErrorString(code), file, line);
        exit(1);
   }
}

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 64>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 64>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

void quant_gemm_cuda(
  void *a_quant,
  void *b_quant,
  void *a_scale,
  void *b_scale,
  void *o_gpu,
  int M,
  int N,
  int K
) {
  using DefaultGemmConf = typename cutlass::gemm::device::DefaultGemmConfiguration<
    OperatorClass,
    SmArch,
    ElementInput,
    ElementInput,
    ElementOutput,
    ElementCompute>;
  using GemmOp = typename DefaultGemmConf::Operator;
  using EpilogueOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput, cutlass::layout::RowMajor,
                                                                  DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
                                                                  ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
                                                                  ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages, true, GemmOp>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<ShapeMMAThreadBlock,
                                                                                               GemmKernel_::kThreadCount, AlphaColTileIterator, typename GemmKernel_::Epilogue::OutputTileIterator,
                                                                                               ElementAccumulator, ElementCompute, EpilogueOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor,
                                                                                                    typename GemmKernel_::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, SwizzleThreadBlock>;

  // // Create a tuple of problem size for matrix multiplication
  // cutlass::gemm::GemmCoord problem_size = { M, N, K };

  // // Initialize alpha and beta for dot product computation
  // ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0);
  // ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

  // // Split K dimension into 1 partitions
  // int split_k_slices = 1;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  typename EpilogueOp::Params linearScalingParams; // TODO: right now it's unused (scaling is done in visitor, no activation needed)
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {reinterpret_cast<ElementInput *>(a_quant), K}, {reinterpret_cast<ElementInput *>(b_quant), K}, {reinterpret_cast<ElementCompute *>(b_scale), 0}, {reinterpret_cast<ElementCompute *>(a_scale), 0}, {nullptr, 0}, {reinterpret_cast<ElementOutput *>(o_gpu), N}, 0, 0, typename EpilogueVisitor::Arguments(linearScalingParams, 0, 0, 0)};

  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  uint8_t *workspace;
  cudaMalloc((void **)&workspace, sizeof(uint8_t) * Gemm::get_workspace_size(arguments));
  checkCudaError();

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);

  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(workspace);
  checkCudaError();
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
  auto A_quant = quantize_tensor(A, -1);
  auto B_quant = quantize_tensor(B, -1);

  int M = A.size(0) * A.size(1);
  int N = B.size(0);
  int K = B.size(1);

  torch::Tensor D = torch::empty({A.size(0), A.size(1), N}, A.options()); // result

  quant_gemm_cuda(
    A_quant.tensor.data_ptr(),
    A_quant.tensor.data_ptr(),
    A_quant.scale.data_ptr(),
    B_quant.scale.data_ptr(),
    D.data_ptr(),
    M,
    N,
    K
  );

  return D;
}