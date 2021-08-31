#include "parrots_cpp_helper.hpp"
using namespace parrots;

namespace pape {
#if (PARROTS_VERSION <= 400)
void gemm(ContextBase& ctx, double alpha, bool tA, DArrayLite A, bool tB, DArrayLite B,
          double beta, DArrayLite C);
#else
void gemm(ContextBase& ctx, double alpha, bool tA, DArrayLite A, bool tB, DArrayLite B,
          double beta, DArrayLite C) {
    parrots::gemm(ctx, alpha, tA, A, tB, B, beta, C);
}
#endif
}
