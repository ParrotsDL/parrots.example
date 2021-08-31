#include "pytorch_cuda_helper.hpp"
using phalf=at::Half;
#include "deform_conv_cuda_kernel.cuh"

using at::Tensor;

void deformable_im2col(
    Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, 
    const int ksize_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    int channel_per_deformable_group = channels / deformable_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_im.type(), "deformable_im2col_cuda", ([&] {
        deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data<scalar_t>(), data_offset.data<scalar_t>(), height, width, ksize_h, ksize_w, pad_h,
            pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, height_col, width_col, data_col.data<scalar_t>());
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
   }
}


void deformable_col2im(
    Tensor data_col,Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, Tensor grad_im) {

    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
    int channel_per_deformable_group = channels / deformable_group;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.type(), "deformable_col2im_cuda", ([&] {
        deformable_col2im_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(), data_offset.data<scalar_t>(), channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, height_col, width_col, grad_im.data<scalar_t>());
    }));
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
     printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
   }
}


void deformable_col2im_coord(
    Tensor data_col, Tensor data_im, Tensor data_offset,
    const int channels, const int height,
    const int width, const int ksize_h,
    const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group,
    Tensor grad_offset) {

    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
    int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data_col.type(), "deformable_col2im_cuda", ([&] {
        deformable_col2im_coord_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(), data_im.data<scalar_t>(), data_offset.data<scalar_t>(), channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, channel_per_deformable_group, height_col, width_col,
            grad_offset.data<scalar_t>());
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
   }
}

