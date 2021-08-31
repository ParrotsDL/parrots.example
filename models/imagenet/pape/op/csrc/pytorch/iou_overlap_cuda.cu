#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include "pytorch_cuda_helper.hpp"
using phalf = at::Half;
#include "iou_overlap_kernel.cuh"

using at::Tensor;
using at::Half;

int IOUOverlap(Tensor bboxes1_data, 
               Tensor bboxes2_data, 
               const int size_bbox,
               const int num_bbox1,
               const int num_bbox2,
               Tensor top_data,
               const int mode,
               const int offset){
    const int kThreadsPerBlock = 1024;
    int output_size = num_bbox1 * num_bbox2;
    //int output_size = num_bbox1;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
            __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bboxes1_data.type(), "IOUOverlap_cuda", ([&] {
        IOUOverlapKernel<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
                     bboxes1_data.data<scalar_t>(), 
                     bboxes2_data.data<scalar_t>(), 
                     size_bbox, num_bbox1, num_bbox2, 
                     top_data.data<scalar_t>(),
                     mode,
                     offset);
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

