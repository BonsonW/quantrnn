#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
    typename Shape_,
    typename WarpMmaOperator_,
    int PartitionsK,
    typename OutputTileIterator_,
    typename AccumulatorFragmentIterator_,
    typename OutputOp_
>
class EpiloguePerRowPerColScaleSimple {
public:
    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using OutputOp = OutputOp_;

    using ElementOutput = typename OutputTileIterator::Element;
    using ElementAccumulator = typename AccumulatorFragmentIterator::Element;
    using ElementCompute = typename OutputOp::ElementCompute;

    using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

    // No shared memory needed
    struct SharedStorage { };

    struct Arguments {
        typename OutputOp::Params output_op;
        ElementCompute const* ptr_row_scale;
        ElementCompute const* ptr_col_scale;
        ElementCompute const* ptr_bias;
        
        CUTLASS_HOST_DEVICE
        Arguments(
            typename OutputOp::Params const& output_op_,
            ElementCompute const* row_scale,
            ElementCompute const* col_scale,
            ElementCompute const* bias = nullptr
        ):
            output_op(output_op_),
            ptr_row_scale(row_scale),
            ptr_col_scale(col_scale),
            ptr_bias(bias) { }
    };

    struct Params {
        typename OutputOp::Params output_op;
        ElementCompute const* ptr_row_scale;
        ElementCompute const* ptr_col_scale;
        ElementCompute const* ptr_bias;
        
        CUTLASS_HOST_DEVICE
        Params(Arguments const& args):
            output_op(args.output_op),
            ptr_row_scale(args.ptr_row_scale),
            ptr_col_scale(args.ptr_col_scale),
            ptr_bias(args.ptr_bias) { }
    };

public:
    CUTLASS_DEVICE
    EpiloguePerRowPerColScaleSimple(
        SharedStorage& shared_storage,
        int thread_idx,
        int warp_idx,
        int lane_idx
    ) { }

    CUTLASS_DEVICE
    void operator()(
        OutputOp const& output_op,
        OutputTileIterator destination_iterator,
        AccumulatorFragment const& accumulators,
        OutputTileIterator source_iterator,
        MatrixCoord const& problem_size,
        MatrixCoord const& threadblock_offset
    ) {
        // Create fragment iterator
        AccumulatorFragmentIterator accum_iterator(accumulators);

        // Process each output element
        typename OutputTileIterator::Fragment output_fragment;
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccumulatorFragment::kElements; ++i) {
            // Get accumulator value
            ElementAccumulator accum_val = accumulators[i];
            
            // Calculate output position
            MatrixCoord coord = destination_iterator.thread_start() + 
                              destination_iterator.threadblock_offset();
            int row_idx = coord.row() + (i / OutputTileIterator::Shape::kColumn);
            int col_idx = coord.column() + (i % OutputTileIterator::Shape::kColumn);
            
            // Load scales with bounds checking
            ElementCompute row_scale = ElementCompute(1);
            ElementCompute col_scale = ElementCompute(1);
            ElementCompute bias = ElementCompute(0);
            
            if (row_idx < problem_size.row() && 
                col_idx < problem_size.column()) {
                
                if (output_op.ptr_row_scale) {
                    row_scale = output_op.ptr_row_scale[row_idx];
                }
                if (output_op.ptr_col_scale) {
                    col_scale = output_op.ptr_col_scale[col_idx];
                }
                if (output_op.ptr_bias) {
                    bias = output_op.ptr_bias[col_idx];
                }
            }
            
            // Apply dequantization
            ElementCompute compute_val = ElementCompute(accum_val);
            compute_val = compute_val * row_scale * col_scale + bias;
            
            // Store result
            output_fragment[i] = ElementOutput(compute_val);
        }
        
        // Write to global memory
        destination_iterator.store(output_fragment);
    }
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/*

// 1. Define the epilogue
using Epilogue = cutlass::epilogue::threadblock::EpiloguePerRowPerColScaleSimple<
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock shape
    WarpMmaOperator,                          // Your warp MMA operator
    1,                                        // PartitionsK
    OutputTileIterator,                       // Output iterator
    AccumulatorFragmentIterator,              // Accumulator iterator
    OutputOp                                  // Your dequant output op
>;

// 2. Set up arguments
typename Epilogue::Arguments epilogue_args(
    output_op_params,
    row_scales_device_ptr,
    col_scales_device_ptr,
    bias_device_ptr  // optional
);

// 3. Create params
typename Epilogue::Params epilogue_params(epilogue_args);

// 4. In your kernel, instantiate and execute
Epilogue epilogue(shared_storage.epilogue, threadIdx.x, warp_idx, lane_idx);

epilogue(
    output_op,
    destination_iterator,
    accumulators,
    source_iterator,
    problem_size,
    threadblock_offset
);

*/