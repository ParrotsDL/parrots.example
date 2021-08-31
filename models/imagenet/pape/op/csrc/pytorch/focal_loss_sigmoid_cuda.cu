#include <ATen/ATen.h>
#include "pytorch_cuda_helper.hpp"
using phalf = at::Half;
#include "focal_loss_sigmoid_kernel.cuh"
using namespace at;


int SigmoidFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses){

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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "sigmoid_focal_loss_forward", ([&] {
      SigmoidFocalLossKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, logits.data<scalar_t>(), 
        targets.data<int>(), 
        weight_pos, gamma, alpha, num_classes, 
        losses.data<scalar_t>());
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
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "sigmoid_focal_loss_backward", ([&] {
      SigmoidFocalLossGradientKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          N, logits.data<scalar_t>(), 
          targets.data<int>(), 
          dX_data.data<scalar_t>(), 
          weight_pos, gamma, alpha, num_classes);
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


