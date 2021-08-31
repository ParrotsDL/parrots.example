#ifndef IOU_OVERLAP_CUH
#define IOU_OVERLAP_CUH

#include <cuda.h>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


template <typename scalar_t>
__global__ void IOUOverlapKernel(
    const scalar_t* bbox1,
    const scalar_t* bbox2,
    const int size_bbox,
    const int num_bbox1,
    const int num_bbox2,
    scalar_t* top_data,
    const int mode,
    const int offset){
    CUDA_KERNEL_LOOP(index, num_bbox1 * num_bbox2){
        int b1 = index / num_bbox2;
        int b2 = index % num_bbox2;

        int base1 = b1 * size_bbox;
        scalar_t b1_x1 = bbox1[base1];
        scalar_t b1_y1 = bbox1[base1 + 1];
        scalar_t b1_x2 = bbox1[base1 + 2];
        scalar_t b1_y2 = bbox1[base1 + 3];
        scalar_t b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset); 

        int base2 = b2 * size_bbox;
        scalar_t b2_x1 = bbox2[base2];
        scalar_t b2_y1 = bbox2[base2 + 1];
        scalar_t b2_x2 = bbox2[base2 + 2];
        scalar_t b2_y2 = bbox2[base2 + 3];
        scalar_t b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset); 

        scalar_t left = fmaxf(b1_x1, b2_x1), right  = fminf(b1_x2, b2_x2);
        scalar_t top  = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
        scalar_t width = fmaxf(right - left + offset, 0.f), height = fmaxf(bottom - top
                                                                   + offset, 0.f);
        scalar_t interS = width * height;
        scalar_t baseS = 1.0;
        if (mode == 0) {
          baseS = fmaxf(b1_area + b2_area - interS, float(offset));
        } else if (mode == 1){ 
          baseS = fmaxf(b1_area, float(offset));
        } else {
          baseS = fmaxf(b2_area, float(offset));
        }   
        top_data[b1 * num_bbox2 + b2] = interS / baseS;
    }
}

#endif
