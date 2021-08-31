#include <math.h>
#include <stdio.h>
#include <float.h>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "iou_overlap_kernel.cuh"

int IOUOverlap(const DArrayLite bboxes1_data,
               const DArrayLite bboxes2_data,
               const int size_bbox,
               const int num_bbox1,
               const int num_bbox2,
               DArrayLite top_data,
               const int mode,
               const int offset){
    const int kThreadsPerBlock = 1024;
    int output_size = num_bbox1 * num_bbox2;
    
    cudaError_t err;
    err = cudaGetLastError();
    if (cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
        bboxes1_data.elemType().prim(), ([&] {
        const scalar_t* bboxes1_ptr = bboxes1_data.ptr<scalar_t>();
        const scalar_t* bboxes2_ptr = bboxes2_data.ptr<scalar_t>();
        scalar_t* top_data_ptr = top_data.ptr<scalar_t>();
        const int blocks = (output_size + kThreadsPerBlock - 1 ) / kThreadsPerBlock;
        IOUOverlapKernel<scalar_t><<<blocks, kThreadsPerBlock>>>(bboxes1_ptr, bboxes2_ptr,
                                                          size_bbox, num_bbox1, num_bbox2,
                                                          top_data_ptr, mode, offset);
        }));           
    err = cudaGetLastError();
    if (cudaSuccess != err){
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    return 1;
}
