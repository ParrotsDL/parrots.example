#include <ATen/ATen.h>
#include "pytorch_cuda_helper.hpp"
#include "psroi_pooling_kernel.cuh"
using namespace at;
using phalf=Half;

int PSROIPoolForwardLaucher(
    Tensor bottom_data, const float spatial_scale, const int num_rois, const int output_dim, 
    const int size_rois, const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, Tensor bottom_rois,
    Tensor top_data, Tensor mapping_channel){
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * output_dim;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.type(), "psroi_pooling_forward_cuda", ([&] {
      PSROIPoolingForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, bottom_data.data<scalar_t>(), spatial_scale, channels, height, width, pooled_height,
        pooled_width, bottom_rois.data<scalar_t>(), output_dim, pooled_height, 
        top_data.data<scalar_t>(), mapping_channel.data<int>(), size_rois);
    }));
    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int PSROIPoolBackwardLaucher(
    Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int output_dim, const int size_rois, const int height, const int width, 
    const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor bottom_diff, Tensor mapping_channel){
    const int kThreadsPerBlock = 1024;
    int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.type(), "psroi_pooling_backward_cuda", ([&] {
      PSROIPoolingBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, top_diff.data<scalar_t>(), mapping_channel.data<int>(), 
        num_rois, spatial_scale, channels, height, width, pooled_height,
        pooled_width, output_dim, bottom_diff.data<scalar_t>(), bottom_rois.data<scalar_t>(), size_rois);
    }));
    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


