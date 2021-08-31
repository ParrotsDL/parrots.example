#ifndef PARROTS_CUDA_HELPER
#define PARROTS_CUDA_HELPER

#include <cuda.h>
#include <parrots/extension.hpp>
#include <parrots/foundation/float16.hpp>
#include <parrots/foundation/mathfunction.hpp>

using namespace parrots;

#define SAFE_DEVICE_CAST(TYPE, x) (TYPE)(x)

#define PARROTS_PRIVATE_CASE_TYPE(prim_type, type, ...) \
    case prim_type: { \
        using scalar_t = type; \
        return __VA_ARGS__(); \
    }

#define PARROTS_DISPATCH_FLOATING_TYPES(TYPE, ...) \
[&] { \
    const auto& the_type = TYPE; \
    switch (the_type) { \
        PARROTS_PRIVATE_CASE_TYPE(Prim::Float64, double, __VA_ARGS__) \
        PARROTS_PRIVATE_CASE_TYPE(Prim::Float32, float, __VA_ARGS__) \
        default: PARROTS_NOTSUPPORTED; \
    }}()

#define PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, ...) \
[&] { \
    const auto& the_type = TYPE; \
    switch (the_type) { \
        PARROTS_PRIVATE_CASE_TYPE(Prim::Float64, double, __VA_ARGS__) \
        PARROTS_PRIVATE_CASE_TYPE(Prim::Float32, float, __VA_ARGS__) \
        PARROTS_PRIVATE_CASE_TYPE(Prim::Float16, float16, __VA_ARGS__) \
        default: PARROTS_NOTSUPPORTED; \
    }}()


/** gemm **/
#if (PARROTS_VERSION <= 400)
// XXX(lizhouyang) Hack to call gemmImpl in the shared library without including the header.
namespace parrots {
namespace compute {
namespace internal {
    #define DECLARE_GEMM_BY_DTYPE(T) \
    template<typename Dev> \
    void gemmImpl(ContextBase& ctx, \
            char transa, char transb, size_t m, size_t n, size_t k, \
            T alpha, const T *A, const T *B, T beta, T *C); \
    \
    template<> \
    void gemmImpl<CudaDevice>(ContextBase&, char, char, size_t, size_t, size_t, \
                  T, const T*, const T*, T, T*);
    
    DECLARE_GEMM_BY_DTYPE(double)
    DECLARE_GEMM_BY_DTYPE(float)
    DECLARE_GEMM_BY_DTYPE(float16)
}  // internal
}  // compute
}  // parrots
#endif

/** atomicAdd **/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

static __inline__ __device__ double atomicAdd(double *address, double val) 
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif


static __inline__ __device__ float16 atomicAdd(float16* address, float16 val) 
{
    unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    unsigned short old_as_us;
    do {
        assumed = old;
        old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);

#if __CUDACC_VER_MAJOR__ >= 9
        float16 tmp;
        tmp.x = old_as_us;
        float16 sum = tmp + val;
        unsigned short sum_as_us = sum.x;
//         half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + (float)(val));
//         unsigned short sum_as_us = __half_as_ushort(sum);
#else
        unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + (float)(val));
#endif

        unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                   : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while(assumed != old);
    //__half_raw raw = {old_as_us};
    //return float16(raw);
    return *reinterpret_cast<float16*>(&old_as_us);
}

#endif // PARROTS_CUDA_HELPER
