#include "nms_kernel.cuh"
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"


void _nms(int boxes_num, const DArrayLite boxes_dev,
          DArrayLite mask_dev, float nms_overlap_thresh, int offset) {

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
    boxes_dev.elemType().prim(), ([&] {
    const scalar_t* boxes_ptr = boxes_dev.ptr<scalar_t>();
    nms_kernel<scalar_t><<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  offset,
                                  boxes_ptr,
                                  (unsigned long long *)mask_dev.ptr<int64_t>());
    }));
}

