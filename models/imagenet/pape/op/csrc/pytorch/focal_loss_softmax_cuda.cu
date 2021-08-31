#include <ATen/ATen.h>
#include "pytorch_cuda_helper.hpp"
using phalf = at::Half;
#include "focal_loss_softmax_kernel.cuh"
using namespace at;

int SoftmaxFocalLossForwardLaucher(
    const int N, Tensor logits,
    Tensor targets, const float weight_pos,
    const float gamma, const float alpha,
    const int num_classes, Tensor losses,
    Tensor priors){

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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "softmax_focal_loss_forward_cuda", ([&] {
      SpatialSoftmaxKernel<scalar_t><<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, logits.data<scalar_t>(), priors.data<scalar_t>(), num_classes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "softmax_focal_forward_cuda", ([&] {
      SoftmaxFocalLossKernel<scalar_t><<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, priors.data<scalar_t>(), targets.data<int>(), losses.data<scalar_t>(), weight_pos, gamma, alpha, num_classes);
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


int SoftmaxFocalLossBackwardLaucher(
    const int N, Tensor logits, Tensor targets,
    Tensor dX_data, const float weight_pos,
    const float gamma, const float alpha, const int num_classes, 
    Tensor priors, Tensor buff){

    const int kThreadsPerBlock = 1024;
    int output_size = N;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "softmax_focal_loss_backward_cuda", ([&] {
      SoftmaxFocalLossGradientWeightKernel<scalar_t><<<(output_size / num_classes + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, priors.data<scalar_t>(), targets.data<int>(), buff.data<scalar_t>(), weight_pos, gamma, alpha, num_classes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.type(), "softmax_focal_backward_cuda", ([&] {
      SoftmaxFocalLossGradientKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        N, priors.data<scalar_t>(), targets.data<int>(), buff.data<scalar_t>(), dX_data.data<scalar_t>(), num_classes);
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


