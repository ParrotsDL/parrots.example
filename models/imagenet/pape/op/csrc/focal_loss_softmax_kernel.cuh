#ifndef FOCAL_LOSS_SOFTMAX_KERNEL_CUH
#define FOCAL_LOSS_SOFTMAX_KERNEL_CUH

#include <stdio.h>
#include <math.h>
#include <float.h>


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void SpatialSoftmaxKernel(const int N, const scalar_t* Xdata, scalar_t* Pdata,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N / num_classes) {
    int base = index * num_classes; //base index

    // Subtract max on each cell for numerical reasons
    scalar_t max_val = -FLT_MAX;
    for(int c = 0; c < num_classes; ++c) {
      max_val = max(max_val, Xdata[base + c]);
    }
    // Exponentiate
    scalar_t expsum = 0.0f;
    for(int c = 0; c < num_classes; ++c) {
      scalar_t expx = expf(Xdata[base + c] - max_val);
      Pdata[base + c] = expx;
      expsum += expx;
    }
    // Normalize
    for(int c = 0; c < num_classes; ++c) {
      Pdata[base + c] /= expsum;
    }
  }
}

template <typename scalar_t>
__global__ void SoftmaxFocalLossKernel(
    const int N, 
    const scalar_t* Pdata, const int* targets, scalar_t* losses,
    const scalar_t weight_pos, const scalar_t gamma, const scalar_t alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N / num_classes) {

    int base = i * num_classes;
    const int label = static_cast<int>(targets[i]);

    scalar_t Np = max(weight_pos, (scalar_t)1.0);
    scalar_t z = (label == 0) * (1 - alpha) / Np +
              (label >= 1) * alpha / Np;

    losses[i] = 0.0;
    if (label >= 0) {
      losses[i] =
          -(powf(1.0 - Pdata[base + label], gamma) *
          log(max(Pdata[base + label], (scalar_t)FLT_MIN))) * z;
    }
  }
}

template <typename scalar_t>
__global__ void SoftmaxFocalLossGradientWeightKernel(
    const int N,
    const scalar_t* Pdata, const int* targets, scalar_t* buff,
    const scalar_t weight_pos, const scalar_t gamma, const scalar_t alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N / num_classes) {

    int base = i * num_classes;
    const int label = static_cast<int>(targets[i]);
    scalar_t Np = max(weight_pos, (scalar_t)1.0);
    scalar_t z =  (label == 0) * (1 - alpha) / Np +
               (label >= 1) * alpha / Np;

    buff[i] = 0.0;
    if (label >= 0) {
      scalar_t onemp = 1. - Pdata[base + label];
      scalar_t p = Pdata[base + label];
      buff[i] =
          (-powf(onemp, gamma) +
          gamma * powf(onemp, gamma - 1) * p * log(max(p, (scalar_t)FLT_MIN))) * z;
    }
  }
}

template <typename scalar_t>
__global__ void SoftmaxFocalLossGradientKernel(
    const int N,
    const scalar_t* Pdata, const int* targets, const scalar_t* buff,
    scalar_t* dX, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N) {

    int ind = i / num_classes;
    int cls = i % num_classes;

    const int label = static_cast<int>(targets[ind]);

    scalar_t c1 = (label >= 0) * 1.0;
    scalar_t c2 = (label == cls) * 1.0;
    dX[i] = 0.0;
    dX[i] = c1 * buff[ind] * (c2 - Pdata[i]);
  }
}
#endif
 
