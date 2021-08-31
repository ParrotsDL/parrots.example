#include "parrots_cuda_helper.hpp"

using namespace parrots;

namespace pape {
#if (PARROTS_VERSION <= 400)
    void gemm(ContextBase& ctx, double alpha, bool tA, DArrayLite A, bool tB, DArrayLite B,
              double beta, DArrayLite C) {
        char transA = tA ? 'T' : 'N';
        char transB = tB ? 'T' : 'N';
        size_t m = tA ? A.dim(1) : A.dim(0);
        size_t n = tB ? B.dim(0) : B.dim(1);
        size_t k = tA ? A.dim(0) : A.dim(1);
        PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(C.elemType().prim(), ([&] {
          parrots::compute::internal::gemmImpl<parrots::CudaDevice>(
            ctx, transB, transA, n, m, k, scalar_t{alpha}, B.ptr<scalar_t>(), A.ptr<scalar_t>(),
            scalar_t{beta}, C.ptr<scalar_t>());
        }));
    }
#endif
}
