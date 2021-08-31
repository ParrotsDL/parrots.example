#include <ATen/ATen.h>
#include <cuda.h>
#include "pytorch_cuda_helper.hpp"
using phalf = at::Half;
#include "roi_align_kernel.cuh"

using at::Tensor;

int ROIAlignForwardLaucher(Tensor bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
                           const int channels, const int aligned_height, const int aligned_width,  const int sampling_ratio,
                           Tensor bottom_rois, Tensor top_data) {
    const int kThreadsPerBlock = 512;
    const int output_size = num_rois * aligned_height * aligned_width * channels;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.type(), "roi_align_forward_cuda", ([&] {
        ROIAlignForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          output_size, 
          bottom_data.data<scalar_t>(), 
          (scalar_t)spatial_scale, height, width, 
          channels, aligned_height, aligned_width, sampling_ratio, 
          bottom_rois.data<scalar_t>(), 
          top_data.data<scalar_t>() );
    }));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int ROIAlignBackwardLaucher(Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
                            const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                            Tensor bottom_rois, Tensor bottom_diff) {
    const int kThreadsPerBlock = 512;
    const int output_size = num_rois * aligned_height * aligned_width * channels;
    cudaError_t err;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.type(), "roi_align_forward_cuda", ([&] {
        ROIAlignBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
          output_size, 
          top_diff.data<scalar_t>(), (scalar_t)spatial_scale, height, width, channels,
          aligned_height, aligned_width,  sampling_ratio, 
          bottom_diff.data<scalar_t>(), 
          bottom_rois.data<scalar_t>()
          );
    }));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
