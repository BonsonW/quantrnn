// Epilogue Output Op for Per-Row-Per-Col Dequantization
// This performs: output = (accumulator * row_scale * col_scale) + bias

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template <
    typename ElementOutput_,           // Output element type (e.g., half_t)
    int Count,                         // Number of elements per thread
    typename ElementAccumulator_,      // Accumulator type (e.g., int32_t or float)
    typename ElementCompute_,          // Computation type (e.g., float)
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class DequantizeEpilogueOp {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    
    using FragmentOutput = cutlass::Array<ElementOutput, Count>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count>;
    using FragmentCompute = cutlass::Array<ElementCompute, Count>;
    
    static int const kCount = Count;
    
    // Parameters that will be passed to each thread
    struct Params {
        ElementCompute const* ptr_row_scale;  // Per-row scales [M]
        ElementCompute const* ptr_col_scale;  // Per-column scales [N]
        ElementCompute const* ptr_bias;       // Optional bias [N]
        
        int64_t row_stride_scale;             // Stride for row scales (usually 1)
        int64_t col_stride_scale;             // Stride for col scales (usually 1)
        int64_t stride_bias;                  // Stride for bias (usually 1)
        
        CUTLASS_HOST_DEVICE
        Params():
            ptr_row_scale(nullptr),
            ptr_col_scale(nullptr), 
            ptr_bias(nullptr),
            row_stride_scale(0),
            col_stride_scale(0),
            stride_bias(0) { }
        
        CUTLASS_HOST_DEVICE
        Params(
            ElementCompute const* row_scale,
            ElementCompute const* col_scale,
            ElementCompute const* bias = nullptr
        ):
            ptr_row_scale(row_scale),
            ptr_col_scale(col_scale),
            ptr_bias(bias),
            row_stride_scale(1),
            col_stride_scale(1),
            stride_bias(1) { }
    };

private:
    Params params_;
    
public:
    CUTLASS_HOST_DEVICE
    DequantizeEpilogueOp(Params const& params): params_(params) { }
    
    // This is the key method - it performs the actual dequantization
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const& accumulator,
        FragmentOutput const& source,           // Usually unused for dequant
        ElementCompute row_scale,               // Scale for this row
        ElementCompute col_scale,               // Scale for this column  
        int row_idx = 0,
        int col_idx = 0
    ) const {
        
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, Count, Round> 
            accumulator_converter;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, Count, Round>
            output_converter;
        cutlass::multiplies<FragmentCompute> mul;
        cutlass::plus<FragmentCompute> add;
        
        // Convert accumulator to compute type
        FragmentCompute converted_acc = accumulator_converter(accumulator);
        
        // Apply row scale: acc * row_scale
        FragmentCompute scaled_by_row = mul(converted_acc, row_scale);
        
        // Apply column scale: (acc * row_scale) * col_scale
        FragmentCompute dequantized = mul(scaled_by_row, col_scale);
        
        // Optional: Add bias if provided
        if (params_.ptr_bias != nullptr) {
            ElementCompute bias_val = params_.ptr_bias[col_idx * params_.stride_bias];
            dequantized = add(dequantized, bias_val);
        }
        
        // Convert back to output type
        return output_converter(dequantized);
    }
    
    // Overload that loads scales from memory
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const& accumulator,
        FragmentOutput const& source,
        int row_idx,
        int col_idx
    ) const {
        // Load the scales for this position
        ElementCompute row_scale = params_.ptr_row_scale[row_idx * params_.row_stride_scale];
        ElementCompute col_scale = params_.ptr_col_scale[col_idx * params_.col_stride_scale];
        
        return (*this)(accumulator, source, row_scale, col_scale, row_idx, col_idx);
    }
};