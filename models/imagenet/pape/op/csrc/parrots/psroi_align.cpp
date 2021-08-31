#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

using namespace parrots;

int PSROIAlignForwardLauncher(
    DArrayLite bottom_data, const float spatial_scale,
    const int num_rois, const int output_dim,
    const int size_rois, const int height,
    const int width, const int channels,
    const int pooled_height, const int pooled_width,
    const float sampling_ratio, DArrayLite bottom_rois,
    DArrayLite top_data, DArrayLite mapping_channel);

int PSROIAlignBackwardLauncher(
    DArrayLite top_diff, const float spatial_scale,
    const int batch_size, const int num_rois,
    const int output_dim, const int size_rois,
    const int height, const int width,
    const int channels, const int pooled_height,
    const int pooled_width,
    const float sampling_ratio, DArrayLite bottom_rois,
    DArrayLite bottom_diff, DArrayLite mapping_channel);


void psroiAlignForwardHost(CudaContext& ctx,
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


void psroiAlignBackwardHost(CudaContext& ctx,
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


int psroiAlignForwardCuda(CudaContext &ctx,
                             const SSElement &attr,
                             const OperatorBase::in_list_t &ins,
                             OperatorBase::out_list_t &outs)
{
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    int sampling_ratio;

    SSAttrs(attr)
        .get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();

    auto &features = ins[0];
    auto &rois = ins[1];

    auto &output = outs[0];
    auto &mapping_channel = outs[1];

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);

    assert(size_rois == 5);

    // data height
    int data_height = features.shape().dim(2);
    // data width
    int data_width = features.shape().dim(3);
    // Number of channels
    int num_channels = features.shape().dim(1);

    PSROIAlignForwardLauncher(
        features, spatial_scale, num_rois, output_dim, size_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        output, mapping_channel);

    return 1;
}

int psroiAlignBackwardCuda(CudaContext &ctx,
                              const SSElement &attr,
                              const OperatorBase::in_list_t &ins,
                              OperatorBase::out_list_t &outs)
{
    int pooled_height;
    int pooled_width;
    int output_dim;
    float spatial_scale;
    int sampling_ratio;

     SSAttrs(attr)
        .get<int>("pooled_height", pooled_height)
        .get<int>("pooled_width", pooled_width)
        .get<int>("output_dim", output_dim)
        .get<float>("spatial_scale", spatial_scale)
        .get<int>("sampling_ratio", sampling_ratio)
        .done();
    
    auto& top_grad = ins[0];
    auto& rois = ins[1];
    auto& mapping_channel = ins[2];

    auto& bottom_grad = outs[0];

    // Number of ROIs
    int num_rois = rois.shape().dim(0);
    int size_rois = rois.shape().dim(1);
    assert(size_rois == 5);

    // batch size
    int batch_size = bottom_grad.shape().dim(0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = bottom_grad.shape().dim(2);
    // data width
    int data_width = bottom_grad.shape().dim(3);
    // Number of channels
    int num_channels = bottom_grad.shape().dim(1);

    PSROIAlignBackwardLauncher(
        top_grad, spatial_scale, batch_size, num_rois, output_dim, size_rois,
        data_height, data_width, num_channels, pooled_height,
        pooled_width, sampling_ratio, rois,
        bottom_grad, mapping_channel);

    return 1;
}

PARROTS_EXTENSION_REGISTER(psroi_align_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("output_dim")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .input(2)
    .output(2)
    .apply(psroiAlignForwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(psroiAlignForwardCuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(psroi_align_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("output_dim")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .input(3)
    .output(1)
    .apply(psroiAlignBackwardHost)
#ifdef PARROTS_USE_CUDA
    .apply(psroiAlignBackwardCuda)
#endif
    .done();
