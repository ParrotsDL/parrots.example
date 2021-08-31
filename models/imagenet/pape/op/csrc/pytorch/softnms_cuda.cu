#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "softnms_kernel.cuh"

using namespace at;

void _softnms(int boxes_num, Tensor boxes_dev, Tensor keep_dev, float sigma,
              float n_thresh, unsigned int method, float overlap_thresh) {

  int boxes_dim = DIVUP(boxes_num, threadsPerBlock);
  dim3 blocks(boxes_dim);
  dim3 threads(threadsPerBlock);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes_dev.type(), "softnms_cuda", ([&] {
    float *max_value = new float[boxes_dim];
    int *max_index = new int[boxes_dim];
    int *order = new int[boxes_num];
    for (int i=0; i < boxes_num; i++){
      order[i] = 0;
    }

    float *max_value_dev;
    int *max_index_dev;
    int *order_dev;
    int size_int = sizeof(int) * boxes_dim;
    int size_float = sizeof(float) * boxes_dim;
    int size_int_boxesnum = sizeof(int) * boxes_num;

    cudaMalloc((void **)&max_value_dev, size_float);
    cudaMalloc((void **)&max_index_dev, size_int);
    cudaMalloc((void **)&order_dev, size_int_boxesnum);

    for (int iter=0; iter<boxes_num; iter++){
      // cudaMemcpy(max_value_dev, max_value, size_float, cudaMemcpyHostToDevice);
      // cudaMemcpy(max_index_dev, max_index, size_int, cudaMemcpyHostToDevice);
      cudaMemcpy(order_dev, order, size_int_boxesnum, cudaMemcpyHostToDevice);
      
      softnms_max_kernel<scalar_t><<<blocks, threads>>>(boxes_num,
                                  overlap_thresh,
                                  boxes_dev.data<scalar_t>(),
                                  order_dev, max_value_dev, max_index_dev);

      cudaMemcpy(max_value, max_value_dev, size_float, cudaMemcpyDeviceToHost);
      cudaMemcpy(max_index, max_index_dev, size_int, cudaMemcpyDeviceToHost);
      cudaMemcpy(order, order_dev, size_int_boxesnum, cudaMemcpyDeviceToHost);
      float max_v = -1.0;
      int max_i = -1;
      for (int b=0; b< boxes_dim; b++){
        if (max_v < max_value[b]){
          max_v = max_value[b];
          max_i = max_index[b];
        }
      }
      if (max_v < 0.0){
        break;
      }
      order[max_i] = boxes_num - iter;
      softnms_update_kernel<scalar_t><<<blocks, threads>>>(boxes_num,
                                  sigma,
                                  n_thresh,
                                  method,
                                  overlap_thresh,
                                  boxes_dev.data<scalar_t>(),
                                  order_dev, (unsigned long long *)keep_dev.data<long>(), max_i);
    }

    cudaFree(max_value_dev);
    cudaFree(max_index_dev);
    cudaFree(order_dev);
  }));
}
