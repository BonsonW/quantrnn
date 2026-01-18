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

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Functions.h>

#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;                        // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;                        // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
  cudaError_t result;

  // quantize and transpose cos its column major
  // ScaledTensor A_scaled = quantize_tensor(_A, 0);
  // ScaledTensor B_scaled = quantize_tensor(_B, -1);
  // _A = _A.permute({1, 0});
  // _B = _B.permute({2, 1, 0});

  int M = A.size(0) * A.size(1);
  int K = B.size(0);
  int N = B.size(1);

  torch::Tensor D = torch::empty({M, N}).to(torch::kHalf).cuda(); // result

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = { M, N, K };

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c( // accumulator
    problem_size.mn());  // <- Create matrix C with dimensions M x N

  // Copy data from host to GPU
  // tensor_a.sync_device();
  // tensor_b.sync_device();
  tensor_c.sync_device();
  // tensor_d.sync_device();
  // tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0.0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  cutlass::half_t* ptr = reinterpret_cast<cutlass::half_t*>(A.data_ptr());

  cutlass::TensorRef<ElementInputA, LayoutInputA> a_ref(
    ptr,
    LayoutInputA(K)   // leading dimension
  );

  ptr = reinterpret_cast<cutlass::half_t*>(B.data_ptr());

  cutlass::TensorRef<ElementInputB, LayoutInputB> b_ref(
    ptr,
    LayoutInputB(N)   // leading dimension
  );

  ptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

  cutlass::TensorRef<ElementOutput, LayoutOutput> d_ref(
    ptr,
    LayoutOutput(N)   // leading dimension
  );

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     a_ref,  // <- reference to matrix A on device
                                     b_ref,  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     d_ref,  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  //
  // Construct events
  //

  cudaEvent_t events[2];

  for (auto & event : events) {
    result = cudaEventCreate(&event);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
      exit(1);
    }
  }

  // Record an event at the start of a series of GEMMs
  result = cudaEventRecord(events[0]);
  if (result != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
    exit(1);
  }

  status = gemm_op();
  CUTLASS_CHECK(status);

  // Record an event when the GEMMs are complete
  result = cudaEventRecord(events[1]);
  if (result != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
    exit(1);
  }

  // Wait for work on the device to complete.
  result = cudaEventSynchronize(events[1]);
  if (result != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
    exit(1);
  }

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  cudaDeviceSynchronize();

  return D;
}