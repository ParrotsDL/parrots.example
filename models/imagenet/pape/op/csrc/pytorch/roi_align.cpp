#include "pytorch_cpp_helper.hpp"
#include <vector>
#include <math.h>
using namespace at;

int ROIAlignBackwardLaucher(
    Tensor top_diff, const float spatial_scale, 
    const int batch_size, const int num_rois, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width, 
    const int sampling_ratio,
    Tensor bottom_rois, Tensor bottom_diff);

int ROIAlignForwardLaucher(
    Tensor bottom_data, const float spatial_scale, 
    const int num_rois, const int height, const int width,
   const int channels, const int aligned_height, const int aligned_width,  
   const int sampling_ratio,
   Tensor bottom_rois, Tensor top_data);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



int roi_align_forward(Tensor features, Tensor rois, Tensor output,
        int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio) {

    if (!features.type().is_cuda()) {
        return 0;
    } else {
        CHECK_INPUT(features);
        CHECK_INPUT(rois);
        CHECK_INPUT(output);

        // Number of ROIs
        int num_rois = rois.size(0);
        int size_rois = rois.size(1);
        if (size_rois != 5) {
            exit(1);
            return 1;
        }

        // data height
        int data_height = features.size(2);
        // data width
        int data_width = features.size(3);
        // Number of channels
        int num_channels = features.size(1);

        ROIAlignForwardLaucher(
            features, spatial_scale, num_rois, data_height,
            data_width, num_channels, aligned_height,
            aligned_width, sampling_ratio, rois,
            output);
    }

    return 0;
}

int roi_align_backward(Tensor top_grad, Tensor rois, Tensor bottom_grad,
        int aligned_height, int aligned_width,
        float spatial_scale, int sampling_ratio) {

    if (!top_grad.type().is_cuda()) {
        AT_ERROR("Not support cpu roi align backward!");
    } else {
        CHECK_INPUT(top_grad);
        CHECK_INPUT(rois);
        CHECK_INPUT(bottom_grad);

        // Number of ROIs
        int num_rois = rois.size(0);
        int size_rois = rois.size(1);
        if (size_rois != 5) {
            exit(1);
            return 1;
        }

        // batch size
        int batch_size = bottom_grad.size(0);
        // data height
        int data_height = bottom_grad.size(2);
        // data width
        int data_width = bottom_grad.size(3);
        // Number of channels
        int num_channels = bottom_grad.size(1);

        ROIAlignBackwardLaucher(
            top_grad, spatial_scale, batch_size, num_rois, data_height,
            data_width, num_channels, aligned_height,
            aligned_width, sampling_ratio, rois,
            bottom_grad);
    }

    return 0;
}
