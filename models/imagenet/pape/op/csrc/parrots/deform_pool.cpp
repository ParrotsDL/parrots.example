#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>
#include "parrots_cpp_helper.hpp"

using namespace parrots;

void deform_psroi_pooling_cpu_forward(HostContext &ctx,
                                       const SSElement &attr,
                                       const OperatorBase::in_list_t &ins,
                                       OperatorBase::out_list_t &outs){
    PARROTS_NOT_IMPL << "Not suppport cpu deformable psroi pooling forward!";
}
void deform_psroi_pooling_cpu_backward(HostContext &ctx,
                                       const SSElement &attr,
                                       const OperatorBase::in_list_t &ins,
                                       OperatorBase::out_list_t &outs){
    PARROTS_NOT_IMPL << "Not suppport cpu deformable psroi pooling backward!";
}
#ifdef PARROTS_USE_CUDA
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
                                const float trans_std);

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
                                    const float trans_std);

void deform_psroi_pooling_cuda_forward(CudaContext &ctx,
                                       const SSElement &attr,
                                       const OperatorBase::in_list_t &ins,
                                       OperatorBase::out_list_t &outs)
{
    int no_trans;
    float spatial_scale;
    int output_dim;
    int group_size;
    int pooled_size;
    int part_size;
    int sample_per_part;
    float trans_std;

    SSAttrs(attr).get<int>("no_trans", no_trans)
                 .get<float>("spatial_scale", spatial_scale)
                 .get<int>("output_dim", output_dim)
                 .get<int>("group_size", group_size)
                 .get<int>("pooled_size", pooled_size)
                 .get<int>("part_size", part_size)
                 .get<int>("sample_per_part", sample_per_part)
                 .get<float>("trans_std", trans_std)
                 .done();

    const auto &input = ins[0];
    const auto &bbox = ins[1];
    const auto &trans = ins[2];
    auto &out = outs[0];
    auto &top_count = outs[1];


    const int batch = input.shape().dim(0);
    const int channels = input.shape().dim(1);
    const int height = input.shape().dim(2);
    const int width = input.shape().dim(3);
    const int channels_trans = no_trans ? 2 : trans.shape().dim(1);

    const int num_bbox = bbox.shape().dim(0);

    DeformablePSROIPoolForward(input, bbox, trans,
                               out, top_count,
                               batch, channels, height, width,
                               num_bbox,
                               channels_trans,
                               no_trans,
                               spatial_scale,
                               output_dim,
                               group_size,
                               pooled_size,
                               part_size,
                               sample_per_part,
                               trans_std);
}

void deform_psroi_pooling_cuda_backward(CudaContext &ctx,
                                       const SSElement &attr,
                                       const OperatorBase::in_list_t &ins,
                                       OperatorBase::out_list_t &outs)
{
    int no_trans;
    float spatial_scale;
    int output_dim;
    int group_size;
    int pooled_size;
    int part_size;
    int sample_per_part;
    float trans_std;
    SSAttrs(attr).get<int>("no_trans", no_trans)
                 .get<float>("spatial_scale", spatial_scale)
                 .get<int>("output_dim", output_dim)
                 .get<int>("group_size", group_size)
                 .get<int>("pooled_size", pooled_size)
                 .get<int>("part_size", part_size)
                 .get<int>("sample_per_part", sample_per_part)
                 .get<float>("trans_std", trans_std)
                 .done();
    
    const auto& out_grad = ins[0];
    const auto& input = ins[1];
    const auto& bbox = ins[2];
    const auto& trans = ins[3];
    const auto& top_count = ins[4];

    auto& input_grad = outs[0];
    auto& trans_grad = outs[1];


    const int batch = input.shape().dim(0);
    const int channels = input.shape().dim(1);
    const int height = input.shape().dim(2);
    const int width = input.shape().dim(3);
    const int channels_trans = no_trans ? 2 : trans.shape().dim(1);

    const int num_bbox = bbox.shape().dim(0);
    

    DeformablePSROIPoolBackwardAcc(out_grad,
                                   input,
                                   bbox,
                                   trans,
                                   top_count,
                                   input_grad,
                                   trans_grad,
                                   batch, channels, height, width, num_bbox,
                                   channels_trans,
                                   no_trans,
                                   spatial_scale,
                                   output_dim,
                                   group_size,
                                   pooled_size,
                                   part_size,
                                   sample_per_part,
                                   trans_std);
}
#endif

PARROTS_EXTENSION_REGISTER(deform_psroi_pooling_cuda_forward)
    .attr("no_trans")
    .attr("spatial_scale")
    .attr("output_dim")
    .attr("group_size")
    .attr("pooled_size")
    .attr("part_size")
    .attr("sample_per_part")
    .attr("trans_std")
    .input(3)
    .output(2)
    .apply(deform_psroi_pooling_cpu_forward)
#ifdef PARROTS_USE_CUDA
    .apply(deform_psroi_pooling_cuda_forward)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(deform_psroi_pooling_cuda_backward)
    .attr("no_trans")
    .attr("spatial_scale")
    .attr("output_dim")
    .attr("group_size")
    .attr("pooled_size")
    .attr("part_size")
    .attr("sample_per_part")
    .attr("trans_std")
   .input(5)
    .output(2)
    .apply(deform_psroi_pooling_cpu_backward)
#ifdef PARROTS_USE_CUDA
    .apply(deform_psroi_pooling_cuda_backward)
#endif
    .done();
