template <
  typename ElementOutput,
  typename ElementAccumulator,
  typename ElementCompute
>
struct Rank1DequantEpilogue {

  struct Params {
    ElementCompute const* inv_sA;  // [M]
    ElementCompute const* inv_sB;  // [N]
  };

  Params params;

  CUTLASS_DEVICE
  Rank1DequantEpilogue(Params const& params_) : params(params_) {}

  CUTLASS_DEVICE
  ElementOutput operator()(
      ElementAccumulator acc,
      int m,
      int n) const {

    ElementCompute x = ElementCompute(acc);
    x *= params.inv_sA[m];
    x *= params.inv_sB[n];
    return ElementOutput(x);
  }
};

// using Epilogue = cutlass::epilogue::threadblock::Epilogue<
//     ShapeMMAThreadBlock,
//     WarpMma,
//     EpilogueOp,
//     LayoutC,
//     ElementsPerAccess
// >;