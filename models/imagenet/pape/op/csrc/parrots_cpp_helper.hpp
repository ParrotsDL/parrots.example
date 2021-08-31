#ifndef PARROTS_CPP_HELPER
#define PARROTS_CPP_HELPER
#include <parrots/foundation/darraylite.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>

using namespace parrots;

namespace pape {
void gemm(ContextBase& ctx, double alpha, bool tA, DArrayLite A, bool tB, DArrayLite B,
          double beta, DArrayLite C);
}

#endif // PARROTS_CPP_HELPER
