#include "pytorch_cpp_helper.hpp"
#include <vector>
#include <math.h>

using at::Tensor;

int PSROIAlignForwardLauncher(
    Tensor bottom_data, const float spatial_scale,
    const int num_rois, const int output_dim,
    const int size_rois, const int height,
    const int width, const int channels,
    const int pooled_height, const int pooled_width,
    const float sampling_ratio, Tensor bottom_rois,
    Tensor top_data, Tensor mapping_channel);

int PSROIAlignBackwardLauncher(
    Tensor top_diff, const float spatial_scale,
    const int batch_size, const int num_rois,
    const int output_dim, const int size_rois,
    const int height, const int width,
    const int channels, const int pooled_height,
    const int pooled_width,
    const float sampling_ratio, Tensor bottom_rois,
    Tensor bottom_diff, Tensor mapping_channel);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int psroi_align_forward_cuda(Tensor features,
                             Tensor rois,
                             Tensor output,
                             Tensor mapping_channel,
                             int pooled_height,
                             int pooled_width,
                             int output_dim,
                             float spatial_scale,
                             int sampling_ratio
                             )
{
    // Grab the input tensor
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mapping_channel);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    assert(size_rois == 5);
    if (size_rois != 5)
    {
        exit(1);
        return 0;
    }

    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    PSROIAlignForwardLauncher(
        features, spatial_scale, num_rois, output_dim, size_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        output, mapping_channel);

    return 1;
}

int psroi_align_backward_cuda(Tensor top_grad,
                              Tensor rois,
                              Tensor mapping_channel,
                              Tensor bottom_grad,
                              int pooled_height,
                              int pooled_width,
                              int output_dim,
                              float spatial_scale,
                              int sampling_ratio)
{
    // Grab the input tensor
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mapping_channel);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    assert(size_rois == 5);
    if (size_rois != 5)
    {
        exit(1);
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    PSROIAlignBackwardLauncher(
        top_grad, spatial_scale, batch_size, num_rois, output_dim, size_rois,
        data_height, data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        bottom_grad, mapping_channel);

    return 1;
}
