#include <ATen/ATen.h>
#include <float.h>
#include <stdio.h>
#include "pytorch_cuda_helper.hpp"
using phalf=at::Half;
#include "roi_pooling_kernel.cuh"

using at::Tensor;

int ROIPoolForwardLaucher(
    Tensor bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor top_data, Tensor argmax_data){
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * channels;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.type(), "roi_pooling_forward_cuda", ([&] {
        ROIPoolForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          output_size, 
          bottom_data.data<scalar_t>(), 
          (scalar_t)spatial_scale, height, width, 
          channels, pooled_height, pooled_width, 
          bottom_rois.data<scalar_t>(), 
          top_data.data<scalar_t>(), 
          argmax_data.data<int>() );
    }));

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


int ROIPoolBackwardLaucher(Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor bottom_diff, Tensor argmax_data){
    const int kThreadsPerBlock = 1024;
    int output_size = batch_size * height * width * channels;
    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.type(), "roi_pooling_backward_cuda", ([&] {
        ROIPoolBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          output_size, 
          top_diff.data<scalar_t>(), 
          argmax_data.data<int>(), 
          num_rois, (scalar_t)spatial_scale, 
          height, width, channels, pooled_height, pooled_width, 
          bottom_diff.data<scalar_t>(), 
          bottom_rois.data<scalar_t>() );
    }));

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
