#include <stdio.h>
#include <float.h>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "roi_pooling_kernel.cuh"

int ROIPoolForwardLaucher(
    const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite top_data, DArrayLite argmax_data, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * channels;
    cudaError_t err;
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        bottom_data.elemType().prim(), ([&] {
            ROIPoolForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              output_size, bottom_data.ptr<scalar_t>(), spatial_scale, height, width, channels, pooled_height,
              pooled_width, bottom_rois.ptr<scalar_t>(), top_data.ptr<scalar_t>(), argmax_data.ptr<int>());
        }));

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int ROIPoolBackwardLaucher(const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite bottom_diff, const DArrayLite argmax_data, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    int output_size = batch_size * height * width * channels;
    cudaError_t err;

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_diff.elemType().prim(), ([&] {
            ROIPoolBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              output_size, top_diff.ptr<scalar_t>(), argmax_data.ptr<int>(), num_rois, spatial_scale, height, width,
              channels, pooled_height, pooled_width, bottom_diff.ptr<scalar_t>(), bottom_rois.ptr<scalar_t>());
        }));

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    return 1;
}

