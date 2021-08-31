// Copyright (c) 2018, SenseTime.

#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>


using namespace parrots; 

void psroiPoolingForwardHost(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .done();
    throw "CPU version of psroi_pooling is not implemented";
}

void psroiPoolingBackwardHost(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .done();
    throw "CPU version of psroi_pooling is not implemented";
}


#ifdef PARROTS_USE_CUDA

int PSROIPoolForwardLaucher(
    const DArrayLite bottom_data, const float spatial_scale, const int num_rois, const int output_dim, const int size_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite top_data, DArrayLite mapping_channel, cudaStream_t stream);


int PSROIPoolBackwardLaucher(
    const DArrayLite top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int output_dim, const int size_rois, const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const DArrayLite bottom_rois,
    DArrayLite bottom_diff, const DArrayLite mapping_channel, cudaStream_t stream);

void psroiPoolingForwardCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .done();
    
    const auto& features = ins[0];
    const auto& rois = ins[1];

    auto& output = outs[0];
    auto& mapping_channel = outs[1];

    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);

    PARROTS_CHECKARGS(size_rois == 5);

    int data_height = features.shape().dim(2);
    int data_width = features.shape().dim(3);
    int num_channels = features.shape().dim(1);

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    PSROIPoolForwardLaucher(
        features, spatial_scale, num_rois, output_dim, size_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois,
        output, mapping_channel, stream);
}

void psroiPoolingBackwardCuda(CudaContext& ctx,
                  const SSElement& attr,
                  const OperatorBase::in_list_t& ins,
                  OperatorBase::out_list_t& outs){
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    SSAttrs(attr).get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .done();

    const auto& top_grad = ins[0];
    const auto& rois = ins[1];
    const auto& mapping_channel = ins[2];

    auto& bottom_grad = outs[0];

    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    PARROTS_CHECKARGS(size_rois == 5);

    int batch_size = bottom_grad.shape().dim(0);
    int data_height = bottom_grad.shape().dim(2);
    int data_width = bottom_grad.shape().dim(3);
    int num_channels = bottom_grad.shape().dim(1);

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());

    PSROIPoolBackwardLaucher(
        top_grad, spatial_scale, batch_size, num_rois, output_dim, size_rois,
        data_height, data_width, num_channels, pooled_height,
        pooled_width, rois,
        bottom_grad, mapping_channel, stream);
}
#endif  // PARROTS_USE_CUDA


PARROTS_EXTENSION_REGISTER(psroi_pooling_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("output_dim")
    .attr("spatial_scale")
    .input(2)
    .output(2)
    .apply(psroiPoolingForwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(psroiPoolingForwardCuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(psroi_pooling_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("output_dim")
    .attr("spatial_scale")
    .input(3)
    .output(1)
    .apply(psroiPoolingBackwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(psroiPoolingBackwardCuda)
#endif
    .done();
