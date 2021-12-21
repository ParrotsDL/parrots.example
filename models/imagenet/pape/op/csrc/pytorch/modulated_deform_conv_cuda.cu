#include <ATen/ATen.h>
#include "pytorch_cuda_helper.hpp"
using phalf=at::Half;
#include "modulated_deform_conv_cuda_kernel.cuh"


void modulated_deformable_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor data_col)
{
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *data_col_ = data_col.data<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_offset_, data_mask_, height_im, width_im, kernel_h, kenerl_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, channels, deformable_group, height_col, width_col, data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_cuda(
    const at::Tensor data_col, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor grad_im)
{

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *grad_im_ = grad_im.data<scalar_t>();

        modulated_deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_offset_, data_mask_, channels, height_im, width_im,
            kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, deformable_group, height_col, width_col, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_coord_cuda(
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    at::Tensor grad_offset, at::Tensor grad_mask)
{
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_, channels, height_im, width_im,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col,
            grad_offset_, grad_mask_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}