#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "psroi_pooling_kernel.cuh"

int PSROIPoolForwardLaucher(
    const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int output_dim, 
    const int size_rois, const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite top_data, DArrayLite mapping_channel, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * output_dim;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        bottom_data.elemType().prim(), ([&] {
          PSROIPoolingForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            output_size, bottom_data.ptr<scalar_t>(), spatial_scale, channels, height, width, pooled_height,
            pooled_width, bottom_rois.ptr<scalar_t>(), output_dim, pooled_height, top_data.ptr<scalar_t>(),
            mapping_channel.ptr<int>(), size_rois);
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
    const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int output_dim, const int size_rois, const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite bottom_diff, const DArrayLite mapping_channel, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;

    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_diff.elemType().prim(), ([&] {
            PSROIPoolingBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              output_size, top_diff.ptr<scalar_t>(), mapping_channel.ptr<int>(), num_rois, spatial_scale, channels,
              height, width, pooled_height, pooled_width, output_dim, bottom_diff.ptr<scalar_t>(), bottom_rois.ptr<scalar_t>(),
              size_rois);
        }));
    err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


