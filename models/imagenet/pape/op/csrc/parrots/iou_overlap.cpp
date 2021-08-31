#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/foundation/exceptions.hpp>
#include <stdio.h>
using namespace parrots;

#define CHECK_CUDA(x) PARROTS_CHECKARGS(int(x.deviceArch()) == 1)
#define CHECK_CPU(x) PARROTS_CHECKARGS(int(x.deviceArch()) == 0)

int IOUOverlap(const DArrayLite bboxes1_data, const DArrayLite bboxes2_data,
               const int size_bbox, const int num_bbox1, const int num_bbox2,
               DArrayLite top_data, const int mode, const int offset);


void cpu_iou_overlaps(HostContext& ctx,
                      const SSElement& attr,
                      const OperatorBase::in_list_t& ins,
                      OperatorBase::out_list_t& outs){
    
    PARROTS_NOT_IMPL;
}

#ifdef PARROTS_USE_CUDA
void gpu_iou_overlaps(CudaContext& ctx,
                      const SSElement& attr,
                      const OperatorBase::in_list_t& ins,
                      OperatorBase::out_list_t& outs){
    int mode, offset;
    SSAttrs(attr)
        .get<int>("mode", mode)
        .get<int>("offset", offset)
        .done();
    
    const auto& bboxes1_data = ins[0];
    const auto& bboxes2_data = ins[1];
    auto& output = outs[0];
    
    // Grad the input tensor
    CHECK_CUDA(bboxes1_data);
    CHECK_CUDA(bboxes2_data);
    CHECK_CUDA(output);

    // Number of boxes
    int num_bbox1 = bboxes1_data.shape().dim(0);
    int num_bbox2 = bboxes2_data.shape().dim(0);
    int size_bbox1 = bboxes1_data.shape().dim(1);
    int size_bbox2 = bboxes2_data.shape().dim(1);
    PARROTS_CHECKARGS(size_bbox1 == size_bbox2);

    // reserve output space
    if (!output){
        output = ctx.createDArrayLite(DArraySpec::array(Prim::Float64, num_bbox1 * num_bbox2), getHostProxy());
    }

   
    IOUOverlap(bboxes1_data, bboxes2_data, size_bbox1, num_bbox1, num_bbox2, output, mode, offset);

}
#endif

PARROTS_EXTENSION_REGISTER(iou_overlap)
    .attr("mode")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(cpu_iou_overlaps)
#ifdef PARROTS_USE_CUDA
    .apply(gpu_iou_overlaps)
#endif
    .done();

