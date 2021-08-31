#include "pytorch_cpp_helper.hpp"
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int IOUOverlap(at::Tensor bboxes1_data, 
               at::Tensor bboxes2_data, 
               const int size_bbox,
               const int num_bbox1,
               const int num_bbox2,
               at::Tensor top_data,
               const int mode,
               const int offset);

using at::Tensor;

void iou_overlaps(Tensor bboxes1, Tensor bboxes2, Tensor output, const int mode, const int offset){
    if (!bboxes1.type().is_cuda() || !bboxes2.type().is_cuda()){
        AT_ERROR("Not support cpu iou overlap!");
    } else{
        // Grad the input tensor
        CHECK_INPUT(bboxes1);
        CHECK_INPUT(bboxes2);
        CHECK_INPUT(output);

        // Number of boxes
        int num_bbox1 = bboxes1.size(0);
        int num_bbox2 = bboxes2.size(0);
        int size_bbox1 = bboxes1.size(1);
        int size_bbox2 = bboxes2.size(1);

        AT_ASSERTM(output.is_cuda(), "output must be cuda tensor");

        AT_ASSERTM(size_bbox1 == size_bbox2, "bbox1 dim must match bbox2");
        
        IOUOverlap(bboxes1,
                   bboxes2,
                   size_bbox1,
                   num_bbox1,
                   num_bbox2,
                   output,
                   mode,
                   offset);
    }
}
