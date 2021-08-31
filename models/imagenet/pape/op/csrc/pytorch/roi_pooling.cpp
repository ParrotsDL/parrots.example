#include "pytorch_cpp_helper.hpp"
#include <vector>

using namespace at;

int ROIPoolForwardLaucher(
    Tensor bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor top_data, Tensor argmax_data);

int ROIPoolBackwardLaucher(Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, Tensor bottom_rois,
    Tensor bottom_diff, Tensor argmax_data);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int roi_pooling_forward(
    Tensor features,
    Tensor rois,
    Tensor output,
    Tensor argmax,
    int pooled_height,
    int pooled_width,
    float spatial_scale){
    if (!features.type().is_cuda()) {
        return 0;
    } else {
        CHECK_INPUT(features);
        CHECK_INPUT(rois);
        CHECK_INPUT(argmax);
        int num_rois = rois.size(0);
        int size_rois = rois.size(1);
        if (size_rois != 5) {
            exit(1);
            return 1;
        }
        int data_height = features.size(2);
        int data_width = features.size(3);
        int num_channels = features.size(1);

        ROIPoolForwardLaucher(
            features, spatial_scale, num_rois, data_height,
            data_width, num_channels, pooled_height,
            pooled_width, rois, output, argmax);
        return 0;
    }
}

int roi_pooling_backward(
    Tensor  top_grad,
    Tensor  rois,
    Tensor  argmax,
    Tensor bottom_grad,
    int pooled_height, 
    int pooled_width, 
    float spatial_scale){
    if (!top_grad.type().is_cuda()) {
        AT_ERROR("Not support cpu roi_pooling backward!");
    } else {
        CHECK_INPUT(top_grad);
        CHECK_INPUT(rois);
        CHECK_INPUT(argmax);
        int num_rois = rois.size(0);
        int size_rois = rois.size(1);
        if (size_rois != 5) {
            exit(1);
            return 1;
        }

        int batch_size = bottom_grad.size(0);
        int num_channels = bottom_grad.size(1);
        int data_height = bottom_grad.size(2); 
        int data_width = bottom_grad.size(3);
        ROIPoolBackwardLaucher(
            top_grad, spatial_scale, 
            batch_size, num_rois, data_height, data_width, 
            num_channels, pooled_height, pooled_width, 
            rois,
            bottom_grad,
            argmax);
        return 0;
     }
}
