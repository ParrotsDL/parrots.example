#include <ATen/ATen.h>
#include "pytorch_cpp_helper.hpp"

using namespace at;

int PSROIMaskPoolForwardLaucher(
    Tensor bottom_data,
    const float spatial_scale, const float roi_scale, const float bin_scale,
    const int num_rois, const int output_dim, const int size_rois,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    Tensor bottom_rois, Tensor top_data, Tensor mapping_channel);


int PSROIMaskPoolBackwardLaucher(
    Tensor top_diff,
    const float spatial_scale, const float roi_scale, const float bin_scale,
    const int batch_size, const int num_rois, const int output_dim,
    const int size_rois, const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    Tensor bottom_rois, Tensor bottom_diff, Tensor mapping_channel);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int psroi_mask_pooling_forward(Tensor features,
            Tensor rois,
            Tensor output,
            Tensor mapping_channel,
            int pooled_height,
            int pooled_width,
            int output_dim,
            float spatial_scale,
            float roi_scale,
            float bin_scale) {
    if (!features.type().is_cuda()) {
        return 0;
    } else { 
        CHECK_INPUT(features);
        CHECK_INPUT(rois);
        CHECK_INPUT(output);
        CHECK_INPUT(mapping_channel);
        int num_rois = rois.size(0);
        int size_rois = rois.size(1);

        AT_ASSERTM(size_rois == 5, "rois shape is expected to be N * 5");
        int feat_height = features.size(2);
        int feat_width = features.size(3);
        int num_channels = features.size(1);

        PSROIMaskPoolForwardLaucher(
            features,
            spatial_scale, roi_scale, bin_scale,
            num_rois, output_dim, size_rois,
            feat_height, feat_width, num_channels,
            pooled_height, pooled_width,
            rois, output, mapping_channel);

        return 0;
    }
}

int psroi_mask_pooling_backward(Tensor top_grad,
             Tensor rois,
             Tensor mapping_channel,
             Tensor bottom_grad,
             int pooled_height,
             int pooled_width,
             int output_dim,
             float spatial_scale,
             float roi_scale,
             float bin_scale) {
    if (!top_grad.type().is_cuda()) {
        return 0;
    } else {
        CHECK_INPUT(top_grad);
        CHECK_INPUT(rois);
        CHECK_INPUT(bottom_grad);
        CHECK_INPUT(mapping_channel);

        int num_rois = rois.size(0);
        int size_rois = rois.size(1);

        AT_ASSERTM(size_rois == 5, "rois shape is expected to be N * 5");
        int batch_size = bottom_grad.size(0);
        int feat_height = bottom_grad.size(2);
        int feat_width = bottom_grad.size(3);
        int num_channels = bottom_grad.size(1);

        PSROIMaskPoolBackwardLaucher(
            top_grad,
            spatial_scale, roi_scale, bin_scale,
            batch_size, num_rois, output_dim, size_rois,
            feat_height, feat_width, num_channels,
            pooled_height, pooled_width,
            rois, bottom_grad, mapping_channel);

        return 0;
    }
}
