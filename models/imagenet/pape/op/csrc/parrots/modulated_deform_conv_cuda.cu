#include <cstdio>
#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "modulated_deform_conv_cuda_kernel.cuh"

using parrots::DArrayLite;

void modulated_deformable_im2col_cuda(ContextBase& ctx,
    const DArrayLite data_im, const DArrayLite data_offset, const DArrayLite data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, DArrayLite data_col)
{
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  auto stream = getStreamNative<CudaDevice>(ctx.getStream());
  // TODO(lizhouyang): enable half.
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_im.elemType().prim(), ([&] {
  modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
    num_kernels, data_im.ptr<scalar_t>(), data_offset.ptr<scalar_t>(), data_mask.ptr<scalar_t>(),
    height_im, width_im, kernel_h, kenerl_w,
    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
    batch_size, channels, deformable_group, height_col, width_col, data_col.ptr<scalar_t>());
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_cuda(ContextBase& ctx,
    const DArrayLite data_col, const DArrayLite data_offset, const DArrayLite data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, DArrayLite grad_im)
{

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  auto stream = getStreamNative<CudaDevice>(ctx.getStream());
  // TODO(lizhouyang): enable half.
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.elemType().prim(), ([&] {
  modulated_deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_col.ptr<scalar_t>(), data_offset.ptr<scalar_t>(), data_mask.ptr<scalar_t>(),
      channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, deformable_group, height_col, width_col, grad_im.ptr<scalar_t>());
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_coord_cuda(ContextBase& ctx,
    const DArrayLite data_col, const DArrayLite data_im, const DArrayLite data_offset, const DArrayLite data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    DArrayLite grad_offset, DArrayLite grad_mask)
{
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;

  auto stream = getStreamNative<CudaDevice>(ctx.getStream());
  // TODO(lizhouyang): enable half.
  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.elemType().prim(), ([&] {
  modulated_deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_col.ptr<scalar_t>(), data_im.ptr<scalar_t>(),
      data_offset.ptr<scalar_t>(), data_mask.ptr<scalar_t>(),
      channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col,
      grad_offset.ptr<scalar_t>(), grad_mask.ptr<scalar_t>());
  }));
      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}
