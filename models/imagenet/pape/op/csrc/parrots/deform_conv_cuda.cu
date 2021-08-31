#include <cstdio>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
#include "deform_conv_cuda_kernel.cuh"


using parrots::DArrayLite;

void deformable_im2col(cudaStream_t stream, DArrayLite data_im,
                       DArrayLite data_offset, const int channels,
                       const int height, const int width, const int ksize_h, 
                       const int ksize_w, const int pad_h, const int pad_w, 
                       const int stride_h, const int stride_w, 
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DArrayLite data_col) {
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_im.elemType().prim(), ([&] {
  deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
    num_kernels, data_im.ptr<scalar_t>(), data_offset.ptr<scalar_t>(),
    height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    channel_per_deformable_group, height_col, width_col, data_col.ptr<scalar_t>());
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


void deformable_col2im(cudaStream_t stream, DArrayLite data_col,
                       DArrayLite data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, DArrayLite grad_im) {

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.elemType().prim(), ([&] {
  deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,stream>>>(
    num_kernels, data_col.ptr<scalar_t>(), data_offset.ptr<scalar_t>(), channels, height, width, ksize_h,
    ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    channel_per_deformable_group, height_col, width_col, grad_im.ptr<scalar_t>());
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


void deformable_col2im_coord(cudaStream_t stream, DArrayLite data_col,
                             DArrayLite data_im, DArrayLite data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group,
                             DArrayLite grad_offset) {

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.elemType().prim(), ([&] {
  deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
    num_kernels, data_col.ptr<scalar_t>(), data_im.ptr<scalar_t>(), data_offset.ptr<scalar_t>(),
    channels, height, width, ksize_h, ksize_w, pad_h,  pad_w, stride_h, stride_w, dilation_h, dilation_w,
    channel_per_deformable_group, height_col, width_col, grad_offset.ptr<scalar_t>());
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

