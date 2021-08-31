#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
#include <stdio.h>
#include <math.h>
#include <float.h>
using phalf=float16;
#include "roi_align_kernel.cuh"

int ROIAlignForwardLaucher(const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
                           const int channels, const int aligned_height, const int aligned_width,  const int sampling_ratio,
                           const DArrayLite bottom_rois, DArrayLite top_data, cudaStream_t stream) {
    const int kThreadsPerBlock = 512;
    const int output_size = num_rois * aligned_height * aligned_width * channels;
    cudaError_t err;

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        bottom_data.elemType().prim(), ([&] {
            ROIAlignForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              output_size, bottom_data.ptr<scalar_t>(), spatial_scale, height, width, channels,
              aligned_height, aligned_width, sampling_ratio, bottom_rois.ptr<scalar_t>(), top_data.ptr<scalar_t>());
        }));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

int ROIAlignBackwardLaucher(const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
                            const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                            const DArrayLite bottom_rois, DArrayLite bottom_diff, cudaStream_t stream) {
    const int kThreadsPerBlock = 512;
    const int output_size = num_rois * aligned_height * aligned_width * channels;
    cudaError_t err;

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_diff.elemType().prim(), ([&] {
            ROIAlignBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              output_size, top_diff.ptr<scalar_t>(), spatial_scale, height, width, channels,
              aligned_height, aligned_width,  sampling_ratio, bottom_diff.ptr<scalar_t>(), bottom_rois.ptr<scalar_t>());
        }));

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
