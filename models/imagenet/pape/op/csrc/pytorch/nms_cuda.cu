#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "nms_kernel.cuh"

using namespace at;

void _nms(int boxes_num, Tensor boxes_dev,
          Tensor mask_dev, float nms_overlap_thresh, int offset) {

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes_dev.type(), "nms_cuda", ([&] {
    nms_kernel<scalar_t><<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  offset,
                                  boxes_dev.data<scalar_t>(),
                                  (unsigned long long *)mask_dev.data<long>());
  }));
}

