#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "focal_loss_sigmoid_kernel.cuh"


int SigmoidFocalLossForwardLaucher(
    const int N, const DArrayLite logits,
    const DArrayLite targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, DArrayLite losses, cudaStream_t stream){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        logits.elemType().prim(), ([&] {
        // Grab the input tensor
        const scalar_t * logits_flat = logits.ptr<scalar_t>();
        const int * targets_flat = targets.ptr<int>();

        scalar_t * losses_flat = losses.ptr<scalar_t>();
        SigmoidFocalLossKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            N, logits_flat, targets_flat, weight_pos, gamma, alpha, num_classes, losses_flat);
        }));
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int SigmoidFocalLossBackwardLaucher(
    const int N, const DArrayLite logits, const DArrayLite targets,
    DArrayLite dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes,
    cudaStream_t stream){

    const int kThreadsPerBlock = 1024;
    int output_size = N;

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        logits.elemType().prim(), ([&] {
    // Grab the input tensor
        const scalar_t * logits_flat = logits.ptr<scalar_t>();
        const int * targets_flat = targets.ptr<int>();

        scalar_t * dX_data_flat = dX_data.ptr<scalar_t>();
        SigmoidFocalLossGradientKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            N, logits_flat, targets_flat, dX_data_flat, weight_pos, gamma, alpha, num_classes);
        }));
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


