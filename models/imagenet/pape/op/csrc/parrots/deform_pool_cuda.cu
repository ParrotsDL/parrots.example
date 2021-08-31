#include <parrots/extension.hpp>
#include "parrots_cuda_helper.hpp"
using phalf=float16;
#include "deform_pool_cuda_kernel.cuh"

void DeformablePSROIPoolForward(const DArrayLite data,
                                const DArrayLite bbox,
                                const DArrayLite trans,
                                DArrayLite out,
                                DArrayLite top_count,
                                const int batch,
                                const int channels,
                                const int height,
                                const int width,
                                const int num_bbox,
                                const int channels_trans,
                                const int no_trans,
                                const float spatial_scale,
                                const int output_dim,
                                const int group_size,
                                const int pooled_size,
                                const int part_size,
                                const int sample_per_part,
                                const float trans_std)
{
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.elemType().prim(), ([&] {
        const scalar_t *bottom_data = data.ptr<scalar_t>();
        const scalar_t *bottom_rois = bbox.ptr<scalar_t>();
        const scalar_t *bottom_trans = no_trans ? NULL : trans.ptr<scalar_t>();
        scalar_t *top_data = out.ptr<scalar_t>();
        scalar_t *top_count_data = top_count.ptr<scalar_t>();

        DeformablePSROIPoolForwardKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, bottom_data, (scalar_t)spatial_scale, channels, height, width, pooled_height, pooled_width,
            bottom_rois, bottom_trans, no_trans, (scalar_t)trans_std, sample_per_part, output_dim,
            group_size, part_size, num_classes, channels_each_class, top_data, top_count_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}

void DeformablePSROIPoolBackwardAcc(const DArrayLite out_grad,
                                    const DArrayLite data,
                                    const DArrayLite bbox,
                                    const DArrayLite trans,
                                    const DArrayLite top_count,
                                    DArrayLite in_grad,
                                    DArrayLite trans_grad,
                                    const int batch,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int num_bbox,
                                    const int channels_trans,
                                    const int no_trans,
                                    const float spatial_scale,
                                    const int output_dim,
                                    const int group_size,
                                    const int pooled_size,
                                    const int part_size,
                                    const int sample_per_part,
                                    const float trans_std)
{
  // LOG(INFO) << "DeformablePSROIPoolBackward";
  const int num_rois = num_bbox;
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  PARROTS_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_grad.elemType().prim(), ([&] {
        const scalar_t *top_diff = out_grad.ptr<scalar_t>();
        const scalar_t *bottom_data = data.ptr<scalar_t>();
        const scalar_t *bottom_rois = bbox.ptr<scalar_t>();
        const scalar_t *bottom_trans = no_trans ? NULL : trans.ptr<scalar_t>();
        scalar_t *bottom_data_diff = in_grad.ptr<scalar_t>();
        scalar_t *bottom_trans_diff = no_trans ? NULL : trans_grad.ptr<scalar_t>();
        const scalar_t *top_count_data = top_count.ptr<scalar_t>();

        DeformablePSROIPoolBackwardAccKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, top_diff, top_count_data, num_rois, (scalar_t)spatial_scale, channels, height, width,
            pooled_height, pooled_width, output_dim, bottom_data_diff, bottom_trans_diff,
            bottom_data, bottom_rois, bottom_trans, no_trans, (scalar_t)trans_std, sample_per_part,
            group_size, part_size, num_classes, channels_each_class);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}
